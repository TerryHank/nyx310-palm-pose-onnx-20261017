#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t,size_t);
extern void *__libc_realloc(void*,size_t);
static __thread int tracing;
static void record(char kind,size_t size) {
    if(size<1024*1024||tracing)return;
    tracing=1;
    char text[96];int n=snprintf(text,sizeof(text),"ALLOC %c %zu\n",kind,size);
    write(2,text,n);tracing=0;
}
void *malloc(size_t size){record('M',size);return __libc_malloc(size);}
void *calloc(size_t count,size_t size){if(size&&count>SIZE_MAX/size){errno=ENOMEM;return NULL;}record('C',count*size);return __libc_calloc(count,size);}
void *realloc(void *p,size_t size){record('R',size);return __libc_realloc(p,size);}
int connect(int fd,const struct sockaddr *address,socklen_t length){(void)fd;(void)address;(void)length;write(2,"NETWORK connect blocked in offline probe\n",41);errno=ECONNREFUSED;return -1;}
int bind(int fd,const struct sockaddr *address,socklen_t length){(void)fd;(void)address;(void)length;errno=EACCES;return -1;}
ssize_t sendto(int fd,const void* data,size_t size,int flags,const struct sockaddr *address,socklen_t length){(void)fd;(void)data;(void)size;(void)flags;(void)address;(void)length;errno=EPERM;return -1;}
int algo_initialize(const void* parameters){
    const char* root=parameters?*(const char*const*)parameters:"<null>";
    write(2,"FACTORY_INIT_ROOT=",18);
    if(root)write(2,root,strnlen(root,4096));
    write(2,"\n",1);
    return -4;
}
