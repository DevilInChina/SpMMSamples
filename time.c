//
// Created by Administrator on 2021/5/18.
//
#include <sys/time.h>
#include <stdlib.h>
#include "my_time.h"
struct timeval t1, t2;
void timeStart(){
    gettimeofday(&t1, NULL);
}
double timeCut(){
    gettimeofday(&t2, NULL);
    return  (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
}
