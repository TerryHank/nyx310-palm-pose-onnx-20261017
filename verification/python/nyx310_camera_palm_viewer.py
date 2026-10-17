"""Same RGB/XZ hand window; all recognition and XYZ come from camera JSON."""
import argparse
import json
import os
import time
import textwrap
from pathlib import Path

import cv2
import numpy as np

from camera_palm_data import CameraPalmReceiver, HAGRID_LABELS, GESTURE_BINARY_MAP, map_grasp_open
from live_ainn_viewer import validate_ipv4
from top_view import compose_view


def render_camera(image, states, fresh, extent, rgb_fps, result_fps, age_ms, error, selected_only=False):
    # Keep the previous layout, fonts, coordinate scale and map implementation.
    # No keypoints, model inference, depth sampling, classification or smoothing.
    display = image.copy()
    cv2.rectangle(display, (0, 0), (display.shape[1], 32), (0, 0, 0), -1)
    age = "--" if age_ms is None else f"{age_ms:.0f}ms"
    cv2.putText(display, f"CAMERA AINN | RGB {rgb_fps:.1f} FPS | AI {result_fps:.1f} FPS | age {age}",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 230, 255), 1)
    map_detections = []
    for state in states:
        if state["xyz_mm"] is None:
            continue
        xyz_m = np.asarray(state["xyz_mm"]) / 1000
        map_detections.append(dict(
            class_name="palm", xyxy=[0, 0, 0, 0], rgb_visible=False,
            position=dict(status="ok" if fresh and state["valid"] else "held",
                          map_xz_m=xyz_m[[0, 2]].tolist(), xyz_m=xyz_m.tolist()),
        ))
    view = compose_view(display, map_detections,
                        dict(status="ok" if fresh else "camera_result_unavailable"),
                        extent=extent, draw_rgb_labels=False,show_target_ids=not selected_only)
    if not states:
        message = "Waiting for camera AINN JSON" if not fresh else (
            "Camera reports no hands")
        if error:
            message = str(error)
        cv2.rectangle(view, (8, 66), (min(1250,view.shape[1]-8), 98), (18,18,18), -1)
        cv2.putText(view, message[:115], (15,90), cv2.FONT_HERSHEY_SIMPLEX,.52,(0,190,255),1)
    for i, state in enumerate(states):
        status = "STALE" if not fresh else "VALID" if state["valid"] else "HOLD" if state["xyz_mm"] else "NO DEPTH"
        prefix='' if selected_only else f"ID {state['hand_id']} | "
        label = f"{prefix}XYZ mm {state['xyz_mm']} | {state['state']} | {status}"
        if 'class_id' in state:
            category = state.get('raw_class_name') or state['state']
            category_label=category if selected_only else f'{HAGRID_LABELS.index(category)} {category}'
            label = f"{prefix}XYZ mm {state['xyz_mm']} | {category_label} {state['confidence']:.2f} | {status}"
        y = 90 + i * 32
        cv2.rectangle(view, (8, y-23), (900, y+6), (18,18,18), -1)
        cv2.putText(view,label,(15,y),cv2.FONT_HERSHEY_SIMPLEX,.52,
                    (70,230,100) if fresh and state["valid"] else (0,190,255),1)
    return view


def render_json_panel(view, packet, status, age_ms, scroll=0):
    """Show the received payload, including old data with explicit STALE status."""
    panel=np.full((292,view.shape[1],3),(22,22,22),np.uint8)
    width=max(40,int((view.shape[1]-32)/7.8))
    if packet is None:
        lines=['No camera JSON received.',str(status.get('error') or 'Waiting for AINN results...')]
        heading='CAMERA JSON | waiting'
    else:
        content=json.dumps(packet['data'],ensure_ascii=False,indent=2)
        lines=[]
        for line in content.splitlines():
            lines.extend(textwrap.wrap(line,width=width,replace_whitespace=False,drop_whitespace=False) or [''])
        fresh=age_ms is not None and age_ms<=500 and status.get('connected') and not status.get('error')
        heading=f"CAMERA JSON | #{packet['result_index']} | camera timestamp {packet['timestamp_ms']} ms | {'LIVE' if fresh else 'STALE'}"
        if packet.get('selected_only'):
            heading=heading.replace('CAMERA JSON', 'CAMERA JSON / GRASP + OPEN')
        if packet.get('source')=='host':
            heading=heading.replace('CAMERA JSON','LOCAL ONNX JSON')
    maximum=max(0,len(lines)-12)
    scroll=max(0,min(int(scroll),maximum))
    cv2.putText(panel,heading,(15,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(70,230,100) if packet else (0,190,255),1)
    cv2.putText(panel,f'Lines {scroll+1}-{min(scroll+12,len(lines))}/{len(lines)} | PgUp/PgDn or mouse wheel | J hide/show',
                (15,46),cv2.FONT_HERSHEY_SIMPLEX,.43,(170,170,170),1)
    for i,line in enumerate(lines[scroll:scroll+12]):
        cv2.putText(panel,line,(15,70+i*18),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1)
    return np.vstack((view,panel)),scroll


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip',type=validate_ipv4,default='192.168.1.101')
    parser.add_argument('--duration-sec',type=float,default=0)
    parser.add_argument('--map-extent',type=float,default=1.5)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-window',action='store_true')
    parser.add_argument('--algorithm',choices=['PalmPoseGraspXYZ','HagridPalmXYZ','HagridBinaryXYZ'],default='PalmPoseGraspXYZ')
    parser.add_argument('--confidence',type=float)
    args=parser.parse_args()
    if args.duration_sec<0 or args.map_extent<=0:parser.error('invalid duration or map extent')
    if args.confidence is not None and not 0 < args.confidence < 1:parser.error('confidence must be between 0 and 1')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'camera_results.jsonl').exists():raise SystemExit(f'output already contains a run: {output}')
    cv2.setNumThreads(1)
    receiver=CameraPalmReceiver(args.ip,args.algorithm,args.confidence)
    native_binary=args.algorithm=='HagridBinaryXYZ'
    selected_only=args.algorithm in ('HagridPalmXYZ','HagridBinaryXYZ')
    title='NYX310 | Camera GRASP / OPEN + Palm XYZ' if native_binary else 'NYX310 | Camera HaGRID + Palm XYZ' if selected_only else 'NYX310 | Palm XYZ mm + OPEN / GRASP / CLOSED'
    frames=0;failure=None;latest=None;started=time.monotonic();last_saved=None
    json_visible=True;json_scroll=0
    try:
        session=receiver.open(output)
        if selected_only and not native_binary:
            session['display_class_mapping']=GESTURE_BINARY_MAP
            session['binary_mapping_policy']='semantic grouping of current detections; ambiguous raw label uses camera stable label or OPEN; no per-hand history'
            session['display_hand_identity']=False
            session['display_schema']='hand_present plus anonymous hands; excludes expired device track records'
            session['display_class_ids']={'GRASP':0,'OPEN':1}
        if native_binary:
            session.update(display_hand_identity=False,native_binary_output=True,
                           binary_mapping_policy='camera native two-class JSON; no host mapping or filtering')
        (output/'session.json').write_text(json.dumps(session,indent=2),encoding='utf-8')
        if not args.no_window:
            cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,1400,880)
            def scroll_json(event,x,y,flags,param):
                nonlocal json_scroll
                if event==cv2.EVENT_MOUSEWHEEL:
                    delta=(flags>>16)&0xffff
                    if delta>=32768:delta-=65536
                    json_scroll=max(0,json_scroll+(-3 if delta>0 else 3))
            cv2.setMouseCallback(title,scroll_json)
        started=time.monotonic()
        while not args.duration_sec or time.monotonic()-started<args.duration_sec:
            if (output/'stop.request').exists():break
            frame=receiver.get_rgb()
            if frame is None:continue
            image,frame_timestamp=frame;frames+=1;now=time.monotonic()
            packet,status=receiver.snapshot()
            display_packet=packet
            if selected_only and packet is not None:
                display_packet=dict(packet,data=packet['data'] if native_binary else map_grasp_open(packet['data']),selected_only=True)
            age_ms=None if packet is None else (now-packet['received_monotonic'])*1000
            fresh=packet is not None and age_ms<=500 and not status['error'] and status['connected']
            states=[] if display_packet is None else display_packet['data']['hands']
            elapsed=max(now-started,1e-9)
            latest=render_camera(image,states,fresh,args.map_extent,frames/elapsed,status['results']/elapsed,age_ms,status['error'],selected_only)
            if json_visible:
                latest,json_scroll=render_json_panel(latest,display_packet,status,age_ms,json_scroll)
            viewer_status=dict(camera=status,source='camera',local_inference=False,
                               frames=frames,result_fresh=fresh,result_age_ms=age_ms,
                               rgb_timestamp_ms=frame_timestamp,
                               result_timestamp_ms=None if packet is None else packet['timestamp_ms'])
            viewer_status.update(json_panel_visible=json_visible,json_panel_scroll=json_scroll)
            if selected_only:
                viewer_status.update(display_classes=[s['state'] for s in states],
                                     camera_hand_count=0 if packet is None else len(packet['data']['hands']),
                                     displayed_hand_count=len(states),
                                     hand_present=bool(states),hand_identity=False,
                                     class_mapping='all current detections -> GRASP / OPEN; expired tracks excluded')
                if native_binary:
                    viewer_status['class_mapping']='camera native GRASP / OPEN; identity tracking removed'
            (output/'viewer_status.json').write_text(json.dumps(viewer_status,indent=2),encoding='utf-8')
            if packet is not None and packet['received_monotonic']!=last_saved:
                last_saved=packet['received_monotonic']
                if selected_only:
                    (output/'display_latest.json').write_text(json.dumps(display_packet['data'],indent=2),encoding='utf-8')
                    with (output/'display_results.jsonl').open('a',encoding='utf-8') as filtered:
                        filtered.write(json.dumps(dict(result_index=packet['result_index'],
                            timestamp_ms=packet['timestamp_ms'],data=display_packet['data']))+'\n')
                if not (output/'first_camera_result.jpg').exists():cv2.imwrite(str(output/'first_camera_result.jpg'),latest)
            if not args.no_window:
                cv2.imshow(title,latest);key=cv2.waitKeyEx(1)
                if frames==1:
                    cv2.setWindowProperty(title,cv2.WND_PROP_TOPMOST,1);cv2.waitKey(1);cv2.setWindowProperty(title,cv2.WND_PROP_TOPMOST,0)
                    (output/'ready.json').write_text(json.dumps(dict(pid=os.getpid(),window_title=title,
                        visible=cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)>=1)),encoding='utf-8')
                    print('FRONTEND_READY: camera RGB + AINN JSON; no host inference or depth computation',flush=True)
                if key in (27,ord('q'),ord('Q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
                if key in (ord('s'),ord('S')):cv2.imwrite(str(output/'snapshot.jpg'),latest)
                if key in (ord('j'),ord('J')):json_visible=not json_visible
                if key in (2162688,65365):json_scroll=max(0,json_scroll-10)
                if key in (2228224,65366):json_scroll+=10
                if key in (2359296,65360):json_scroll=0
    except KeyboardInterrupt:
        pass
    except Exception as error:
        failure=f'{type(error).__name__}: {error}';print(f'ERROR: {failure}',flush=True)
    finally:
        receiver.close();cv2.destroyAllWindows()
        if latest is not None:cv2.imwrite(str(output/'last_display.jpg'),latest)
    _,status=receiver.snapshot();elapsed=max(time.monotonic()-started,1e-9)
    summary=dict(elapsed_sec=round(elapsed,3),frames=frames,rgb_fps=round(frames/elapsed,2),
                 ainn_results=status['results'],camera_status=status,local_inference=False,
                 host_depth_computation=False,host_classification=False,
                 camera_test_executed=frames>0,camera_json_received=status['results']>0,
                 failure=failure,close_statuses=receiver.close_statuses)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if failure:return 2
    if not status['results']:
        print('WARNING: RGB only; no validated camera hand JSON received. No host fallback was used.',flush=True)
        return 3
    return 0


if __name__=='__main__':raise SystemExit(main())
