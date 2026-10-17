"""PC OpenCV Zoo Palm + Hand Pose, ROI/Kalman, OPEN/PINCH/GRIP and ToF."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from live_ainn_viewer import validate_ipv4
from nyx310_rgbd import Nyx310RgbdCamera
from nyx310_camera_palm_viewer import render_camera,render_json_panel
from palm_position import measure_palm,palm_geometry
from palm_pose_pc import PalmRoiTracker,SinglePalmState,classify_hand


def draw_hand_overlay(image,hand,current):
    display=image.copy()
    if hand is None or current is None:
        return display
    xy=np.rint(hand['points'][:,:2]).astype(np.int32)
    chains=((0,1,2,3,4),(0,5,6,7,8),(5,9,10,11,12),
            (9,13,14,15,16),(13,17,18,19,20),(0,17))
    for chain in chains:
        cv2.polylines(display,[xy[list(chain)]],False,(60,230,90),2,cv2.LINE_AA)
    for point in xy:
        cv2.circle(display,tuple(point),3,(0,220,255),-1,cv2.LINE_AA)
    reference,polygon=palm_geometry(hand['points'])
    cv2.polylines(display,[np.rint(polygon).astype(np.int32)],True,(255,210,50),2,cv2.LINE_AA)
    cv2.drawMarker(display,tuple(np.rint(reference).astype(int)),(255,255,255),cv2.MARKER_CROSS,17,2)
    height,width=image.shape[:2]
    pad=max(8,int(np.max(np.ptp(xy,axis=0))*.08))
    x1,y1=np.maximum(0,xy.min(axis=0)-pad)
    x2,y2=np.minimum([width-1,height-1],xy.max(axis=0)+pad)
    cv2.rectangle(display,(int(x1),int(y1)),(int(x2),int(y2)),(60,230,90),2)
    label=f"{current['state']} {hand['confidence']:.2f}"
    (text_width,text_height),baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.65,2)
    left=max(0,min(int(x1),width-text_width-8))
    bottom=int(y1)-6 if y1>text_height+38 else min(height-baseline-3,int(y1)+text_height+8)
    cv2.rectangle(display,(left,bottom-text_height-5),(min(width-1,left+text_width+7),bottom+baseline+3),(25,25,25),-1)
    cv2.putText(display,label,(left+3,bottom),cv2.FONT_HERSHEY_SIMPLEX,.65,(60,230,90),2,cv2.LINE_AA)
    return display


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip',type=validate_ipv4,default='192.168.1.101')
    parser.add_argument('--duration-sec',type=float,default=0)
    parser.add_argument('--map-extent',type=float,default=1.5)
    parser.add_argument('--palm-confidence',type=float,default=.55)
    parser.add_argument('--pose-confidence',type=float,default=.6)
    parser.add_argument('--rgb-exposure-cap-us',type=int,default=8000)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-window',action='store_true')
    parser.add_argument('--record-rgb',action='store_true')
    args=parser.parse_args()
    if args.duration_sec<0 or args.map_extent<=0 or not 0<args.palm_confidence<1 or not 0<args.pose_confidence<1:
        parser.error('invalid duration, map extent or confidence')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'session.json').exists():raise SystemExit(f'output already contains a run: {output}')
    cv2.setNumThreads(1)
    tracker=PalmRoiTracker(args.palm_confidence,args.pose_confidence)
    state=SinglePalmState()
    camera=None;video=None;writer=None;image_source=None;failure=None;latest=None
    title='NYX310 | PC Palm + Hand Pose | OPEN / PINCH / GRIP'
    frames=observations=valid=matched=timeouts=0
    counts=Counter();reasons=Counter();model_ms=[]
    started=time.monotonic();show_json=True;scroll=0
    session=dict(backend='PC OpenCV DNN CPU',camera_ai_api_used=False,local_inference=True,
                 host_classification=True,host_depth_computation=True,hand_identity=False,
                 models=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                         for p in (tracker.palm_path,tracker.pose_path)],
                 palm_confidence=args.palm_confidence,pose_confidence=args.pose_confidence,
                 classes=['OPEN','PINCH','GRIP'],roi_policy='Palm on acquisition/reacquisition; Hand Pose on Kalman-predicted ROI; predict up to250ms before full detection',
                 reference='(wrist0+middle_mcp9)/2; contracted palm ROI median + MAD + ToF calibration',
                 xyz_coordinates='RGB camera X right,Y down,Z forward; mm',
                 prediction_is_measurement=False,rgb_exposure_cap_policy='temporary auto-exposure ceiling, restored on exit',
                 display_overlay='21 landmarks; gesture bounding box; cyan palm depth ROI and white fixed reference point')
    try:
        if args.source:
            session['source']=str(args.source.resolve())
            if args.source.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.webp'):
                image_source=cv2.imread(str(args.source.resolve()))
                if image_source is None:raise RuntimeError('Cannot read source image')
                fps=15.
            else:
                video=cv2.VideoCapture(str(args.source.resolve()))
                if not video.isOpened():raise RuntimeError('Cannot open source video')
                fps=video.get(cv2.CAP_PROP_FPS) or 15.
        else:
            camera=Nyx310RgbdCamera(args.ip,allow_discovery=True,capture_profile=True,rgb_exposure_cap_us=args.rgb_exposure_cap_us)
            session['camera']=camera.open()
        (output/'session.json').write_text(json.dumps(session,indent=2),encoding='utf-8')
        if not args.no_window:
            cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,1400,880)
        started=time.monotonic();best_score=(-1,-1)
        with (output/'results.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            while not args.duration_sec or time.monotonic()-started<args.duration_sec:
                if (output/'stop.request').exists():break
                packet=None
                if camera:
                    packet=camera.get_frame()
                    if packet is None:timeouts+=1;continue
                    image=packet['bgr'];stamp=packet['color_timestamp_us']//1000
                    now=stamp/1000;matched+=int(packet['timestamp_match'])
                elif image_source is not None:
                    if frames:break
                    image=image_source;now=0.;stamp=0
                else:
                    ok,image=video.read()
                    if not ok:break
                    now=frames/fps;stamp=round(now*1000)
                if args.record_rgb and camera:
                    if writer is None:
                        writer=cv2.VideoWriter(str(output/'rgb_replay.avi'),cv2.VideoWriter_fourcc(*'MJPG'),15.,(image.shape[1],image.shape[0]))
                        if not writer.isOpened():raise RuntimeError('Cannot open RGB replay writer')
                    writer.write(image)
                frames+=1;t0=time.perf_counter()
                hand=tracker.infer(image,now)
                infer_ms=(time.perf_counter()-t0)*1000;model_ms.append(infer_ms)
                xyz=None;gesture=None;geometry={};depth_info=dict(reason='no_current_landmarks')
                if hand is not None:
                    gesture,geometry=classify_hand(hand['world'],hand['points'])
                    if gesture is None:hand=None
                if hand is not None:
                    observations+=1;counts[gesture]+=1
                    if packet is not None:
                        xyz,depth_info=measure_palm(hand['points'],packet['depth_mm'],camera.color_K,camera.color_D,packet['timestamp_match'])
                    else:depth_info=dict(reason='offline_rgb_has_no_tof')
                current=state.update(hand,xyz,gesture,now)
                hands=[] if current is None else [current]
                valid+=int(current is not None and current['valid']);reasons[depth_info['reason']]+=1
                data=dict(hand_present=hand is not None,hands=hands,tracking_status=tracker.status)
                record=dict(frame_index=frames,timestamp_ms=stamp,source='host',data=data,
                            inference_ms=round(infer_ms,3),palm_calls=tracker.palm_calls,
                            pose_calls=tracker.pose_calls,kalman_predictions=tracker.predictions,
                            rule_features=geometry,depth=depth_info,
                            roi=None if hand is None else hand['roi_small'])
                log.write(json.dumps(record)+'\n')
                (output/'latest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
                elapsed=max(time.monotonic()-started,1e-9)
                display=draw_hand_overlay(image,hand,current)
                latest=render_camera(display,hands,True,args.map_extent,frames/elapsed,frames/elapsed,None,None,True)
                cv2.rectangle(latest,(0,0),(min(970,latest.shape[1]),32),(0,0,0),-1)
                cv2.putText(latest,f'LOCAL ONNX | {tracker.status} | {frames/elapsed:.1f} FPS | {infer_ms:.0f}ms | OPEN / PINCH / GRIP',
                            (8,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,230,255),1)
                if show_json:
                    latest,scroll=render_json_panel(latest,dict(data=data,result_index=frames,timestamp_ms=stamp,source='host'),
                                                   dict(connected=True,error=None),0,scroll)
                score=(int(current is not None and current['valid']),hand['confidence'] if hand else 0.)
                if score>best_score:
                    best_score=score;cv2.imwrite(str(output/'best_display.jpg'),latest);cv2.imwrite(str(output/'best_rgb.jpg'),image)
                if packet is not None and current is not None and current['valid'] and not (output/'valid_depth_sample.npz').exists():
                    np.savez_compressed(output/'valid_depth_sample.npz',bgr=image,depth_mm=packet['depth_mm'],
                                        points=hand['points'],world=hand['world'],K=camera.color_K,D=camera.color_D)
                status=dict(pid=os.getpid(),frames=frames,fps=round(frames/elapsed,2),backend='host_onnx',
                            local_inference=True,camera_ai_api_used=False,hand_present=data['hand_present'],
                            current_states=[h['state'] for h in hands],valid_positions=valid,
                            tracking_status=tracker.status,palm_calls=tracker.palm_calls,pose_calls=tracker.pose_calls,
                            kalman_predictions=tracker.predictions)
                (output/'viewer_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
                if not args.no_window:
                    cv2.imshow(title,latest);key=cv2.waitKeyEx(1)
                    if frames==1:
                        cv2.setWindowProperty(title,cv2.WND_PROP_TOPMOST,1);cv2.waitKey(1);cv2.setWindowProperty(title,cv2.WND_PROP_TOPMOST,0)
                        (output/'ready.json').write_text(json.dumps(dict(pid=os.getpid(),window_title=title,
                            visible=cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)>=1)),encoding='utf-8')
                        print('FRONTEND_READY: PC Palm + Hand Pose + ToF; OPEN / PINCH / GRIP',flush=True)
                    if key in (27,ord('q'),ord('Q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
                    if key in (ord('s'),ord('S')):cv2.imwrite(str(output/'snapshot.jpg'),latest)
                    if key in (ord('j'),ord('J')):show_json=not show_json
                    if key in (2162688,65365):scroll=max(0,scroll-10)
                    if key in (2228224,65366):scroll+=10
    except KeyboardInterrupt:
        pass
    except Exception as error:
        failure=f'{type(error).__name__}: {error}';print('ERROR: '+failure,flush=True)
    finally:
        if camera:camera.close()
        if video:video.release()
        if writer:writer.release()
        cv2.destroyAllWindows()
        if latest is not None:cv2.imwrite(str(output/'last_display.jpg'),latest)
    elapsed=max(time.monotonic()-started,1e-9)
    summary=dict(elapsed_sec=round(elapsed,3),frames=frames,fps=round(frames/elapsed,2),
                 hand_observations=observations,valid_positions=valid,matched_depth_frames=matched,
                 frame_timeouts=timeouts,gestures=dict(counts),depth_reasons=dict(reasons),
                 palm_calls=tracker.palm_calls,pose_calls=tracker.pose_calls,kalman_predictions=tracker.predictions,
                 average_model_ms=float(np.mean(model_ms)) if model_ms else None,
                 failure=failure,local_inference=True,camera_ai_api_used=False,
                 camera_test_executed=camera is not None and frames>0,
                 close_statuses=[] if camera is None else camera.close_statuses)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return 2 if failure or not frames else 0


if __name__=='__main__':raise SystemExit(main())
