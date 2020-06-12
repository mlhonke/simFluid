//
// Created by graphics on 19/01/20.
//
#ifndef FERRO3D_EXECTIMER_HPP
#define FERRO3D_EXECTIMER_HPP

#include <iostream>
#include <time.h>
#include <string>

class ExecTimer {
public:
    ExecTimer(char const* message) : message(message) {
        clock_gettime(CLOCK_MONOTONIC, &begin);
    }

    ~ExecTimer(){
        clock_gettime(CLOCK_MONOTONIC, &end);
        auto time_spent = (double)(end.tv_sec-begin.tv_sec);
        time_spent += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        std::cout << message << " : " << time_spent << std::endl;
    }

protected:
    struct timespec begin, end;
    std::string message;

private:
};

class ExecTimerCumulative : public ExecTimer{
public:
    ExecTimerCumulative(char const* message) : ExecTimer(message) {
        clock_gettime(CLOCK_MONOTONIC, &begin_lap);
    }

    void lap() {
        laps++;
        clock_gettime(CLOCK_MONOTONIC, &end_lap);
        auto time_spent = (double)(end_lap.tv_sec-begin_lap.tv_sec);
        time_spent += (end_lap.tv_nsec - begin_lap.tv_nsec) / 1000000000.0;
        total += time_spent;
        std::cout << "Average time per step: " << total / (double) laps << std::endl;
        clock_gettime(CLOCK_MONOTONIC, &begin_lap);
    }

private:
    double total = 0;
    int laps = 0;
    struct timespec begin_lap, end_lap;
};

#endif //FERRO3D_EXECTIMER_HPP
