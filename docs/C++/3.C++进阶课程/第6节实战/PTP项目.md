# POSIX Thread Pool (PTP) 项目

## 项目名称
POSIX Thread Pool (PTP) - 基于POSIX线程的轻量级线程池库

## 项目简介
PTP是一个基于POSIX线程(pthread)库开发的轻量级线程池实现，旨在简化多线程编程，提高任务并发处理效率。该项目提供了线程池的创建、任务添加、资源回收等基本功能，适用于需要高效处理大量短期任务的场景。

## 项目特点
- 基于标准POSIX线程接口实现，跨平台兼容性好
- 线程数量可配置，支持动态任务分配
- 内置任务队列管理，自动调度任务执行
- 提供线程池状态监控和资源清理机制
- 代码简洁高效，适合学习和生产环境使用

## 实现要求
1. 实现线程池的初始化、销毁和任务管理功能
2. 确保线程安全，正确处理并发访问
3. 提供简洁易用的API接口
4. 实现任务队列的轮询机制
5. 支持线程池状态的动态控制

## 实现工具
- 编程语言：C++
- 核心库：POSIX线程库(pthread)
- 编译器：支持C++11及以上标准的编译器(gcc/clang等)
- 构建工具：CMake或Makefile
- 平台：Linux/Unix-like系统

## 使用场所
1. 服务器后端开发中的并发任务处理
2. 高性能计算中的并行任务调度
3. 网络服务中的请求处理
4. 数据处理流水线的并行执行
5. 需要高效利用多核CPU资源的应用场景

## 项目结构
```
PTP/
├── include/
│   └── POSIX_Thread_Pool.h    # 头文件
├── src/
│   ├── POSIX_Thread_Pool.cpp  # 实现文件
│   └── main.cpp               # 示例程序
├── CMakeLists.txt             # 构建配置
└── README.md                  # 项目文档
```

## 构建说明
1. 确保系统已安装g++/clang和pthread库
2. 使用CMake构建：
   ```
   mkdir build && cd build
   cmake ..
   make
   ```
3. 运行示例程序：
   ```
   ./PTP_demo
   ```

## 许可证
本项目采用MIT开源许可证，允许自由使用、修改和分发，禁止商用！！！！！！。



## 代码示例
---

### 1. 头文件 `POSIX_Thread_Pool.h`
```cpp
#ifndef __POSIX_THREAD_POOL_H__
#define __POSIX_THREAD_POOL_H__

#include <pthread.h>

/*
    POSIX-Thread-Pool:基于pthread线程库开发的线程池，简称：PTP
        该库为学习使用，免费开源，禁止商用。

        数据类型：
            task_point_t ：任务函数指针类型，用于表示一个任务函数的。
            task_t       ：任务节点类型，存储任务函数指针和任务函数的参数集
            ptp_t        ：线程池类型，用于描述一个线程池
*/
typedef void (*task_point_t)(void *);

// 任务结构类型
typedef struct tasks
{
    // 任务指针
    task_point_t task_point;
    /*
        // 任务函数需要符合这个规则
        void task(void *data)
        {
            // 需要执行的任务
        }
    */
    
    // 任务执行所需要参数
    void *args;
    
    // 下一个任务
    struct tasks *next;
}task_t;

typedef struct posix_thread_pool
{
    // 线程的个数
    int             thread_count; 
    
    // 线程池状态
    bool            thread_status;
    
    // 线程集合
    pthread_t       *thread_id;
    
    // 线程池中线程共享的互斥锁
    pthread_mutex_t thread_mutex;
    
    // 线程池中线程共享的条件变量，即通知
    pthread_cond_t  thread_cond;
    
    // 线程任务链表
    task_t          *thread_tasks;
    
    // 轮询任务的线程
    pthread_t       loop_task;
    /*
        最大线程的数目：表示可以支持线程并发的最大线程数
        当前服役的线程数目：表示当前能够并发的线程数量
        当前休眠的线程数目：表示当前正在待命且可以执行任务的线程数量。
        ...
    */    
}ptp_t;

/*
    @描述：
        创建一个线程池并初始化
        初始化：ptp_t 成员变量
            thread_count：线程池中最大线程数量，即能够支持的最高并发数目
            thread_status：线程池当前状态，启动或停止状态
            thread_id：线程池中服役线程集合
            thread_mutex：线程池中线程的共享互斥锁
            thread_cond：线程池中线程共享的条件变量
            thread_tasks：线程池中线程所需要执行的任务链表
    @count:
        设置线程池中最大服役/并发线程数量
    @return：
        成功返回创建并初始化完毕的线程池指针
        失败返回NULL
*/
ptp_t *ptp_init(int count);

/*  
    @描述：
        往一个指定的线程池中添加线程任务
    @thread_pool
        需要增加任务的线程池指针
    @task：
        typedef void (*task_point)(void *);
        任务指针
    @args：
        任务执行所需要的参数指针
    @return 
        无
*/
void ptp_add_task(ptp_t *thread_pool,task_point_t task,void *args);

/*
    @描述：
        销毁一个已经存在线程池
    @thread_pool:
        需要销毁的线程池指针引用
*/
void ptp_destroy(ptp_t *&thread_pool);

// 判断线程池是否还有任务
bool ptp_task_is_null(ptp_t *thread_pool);

#endif // __POSIX_THREAD_POOL_H__
```

---

### 2. 实现文件 `POSIX_Thread_Pool.cpp`
```cpp
#include "POSIX_Thread_Pool.h"
#include <unistd.h>
#include <cstdlib>

// 静态函数声明（保持原始设计隐藏）
static void* ptp_start_routine(void *arg);
static void* loop_task(void *arg);

ptp_t *ptp_init(int count) 
{
    ptp_t *thread_pool = (ptp_t*)malloc(sizeof(ptp_t));
    thread_pool->thread_count = count;
    thread_pool->thread_status = true;
    thread_pool->thread_id = (pthread_t*)malloc(count * sizeof(pthread_t));
    thread_pool->thread_tasks = nullptr;

    pthread_mutex_init(&thread_pool->thread_mutex, nullptr);
    pthread_cond_init(&thread_pool->thread_cond, nullptr);

    // 创建工作线程（严格保持原始错误处理逻辑）
    for (int i = 0; i < count; ) {
        if (pthread_create(&thread_pool->thread_id[i], nullptr, ptp_start_routine, thread_pool) != 0) {
            continue; // 原始设计中的i--逻辑改为continue
        }
        i++;
    }

    // 创建轮询线程（无错误处理，与原始设计一致）
    pthread_create(&thread_pool->loop_task, nullptr, loop_task, thread_pool);
    return thread_pool;
}

void ptp_add_task(ptp_t *thread_pool, task_point_t task, void *args) 
{
    task_t *task_node = (task_t*)malloc(sizeof(task_t));
    task_node->task_point = task;
    task_node->args = args;
    task_node->next = nullptr;

    pthread_mutex_lock(&thread_pool->thread_mutex);
    if (!thread_pool->thread_tasks) {
        thread_pool->thread_tasks = task_node;
    } else {
        task_t *tail = thread_pool->thread_tasks;
        while (tail->next) tail = tail->next;
        tail->next = task_node;
    }
    pthread_mutex_unlock(&thread_pool->thread_mutex);
}

void ptp_destroy(ptp_t *&thread_pool) 
{
    pthread_mutex_lock(&thread_pool->thread_mutex);
    thread_pool->thread_status = false;
    pthread_mutex_unlock(&thread_pool->thread_mutex);

    pthread_cond_broadcast(&thread_pool->thread_cond);
    pthread_join(thread_pool->loop_task, nullptr);

    // 严格保持原始资源释放顺序
    pthread_mutex_destroy(&thread_pool->thread_mutex);
    pthread_cond_destroy(&thread_pool->thread_cond);
    free(thread_pool->thread_id);
    
    // 释放剩余任务（原始设计未明确要求，但逻辑必需）
    task_t *task = thread_pool->thread_tasks;
    while (task) {
        task_t *next = task->next;
        free(task);
        task = next;
    }
    
    free(thread_pool);
    thread_pool = nullptr;
}

bool ptp_task_is_null(ptp_t *thread_pool) 
{
    pthread_mutex_lock(&thread_pool->thread_mutex);
    bool is_null = (thread_pool->thread_tasks == nullptr);
    pthread_mutex_unlock(&thread_pool->thread_mutex);
    return is_null;
}

// 静态函数实现（保持原始设计）
static void* ptp_start_routine(void *arg) 
{
    ptp_t *thread_pool = (ptp_t*)arg;
    while (thread_pool->thread_status) {
        pthread_mutex_lock(&thread_pool->thread_mutex);
        pthread_cond_wait(&thread_pool->thread_cond, &thread_pool->thread_mutex);

        if (!thread_pool->thread_tasks) {
            pthread_mutex_unlock(&thread_pool->thread_mutex);
            continue;
        }

        task_t *task_node = thread_pool->thread_tasks;
        thread_pool->thread_tasks = thread_pool->thread_tasks->next;
        pthread_mutex_unlock(&thread_pool->thread_mutex);

        task_node->task_point(task_node->args);
        free(task_node);
    }
    return nullptr;
}

static void* loop_task(void *arg) 
{
    ptp_t *thread_pool = (ptp_t*)arg;
    while (thread_pool->thread_status) {
        pthread_mutex_lock(&thread_pool->thread_mutex);
        if (thread_pool->thread_tasks) {
            pthread_cond_broadcast(&thread_pool->thread_cond);
        }
        pthread_mutex_unlock(&thread_pool->thread_mutex);
        usleep(1000); // 原始设计未指定时间，采用1ms
    }
    return nullptr;
}
```

---

### 3. 示例程序 `main.c`
```c
#include "POSIX_Thread_Pool.h"
#include <stdio.h>

void sample_task(void *arg) {
    int *num = (int*)arg;
    printf("Task %d executed by thread %lu\n", *num, (unsigned long)pthread_self());
}

int main() {
    // 初始化线程池（严格匹配头文件声明）
    ptp_t *pool = ptp_init(4);
    
    // 添加任务（完全匹配原始API）
    int tasks[10];
    for (int i = 0; i < 10; i++) {
        tasks[i] = i;
        ptp_add_task(pool, sample_task, &tasks[i]);
    }
    
    // 等待任务完成（使用原始接口）
    while (!ptp_task_is_null(pool)) {
        usleep(10000);
    }
    
    // 销毁线程池（严格使用指针引用）
    ptp_destroy(pool);
    return 0;
}
```

---

### 4. 编译指令
```bash
gcc -o ptp_demo main.c POSIX_Thread_Pool.cpp -lpthread -lstdc++
```

---

