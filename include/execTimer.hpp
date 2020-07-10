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
        double time_spent = time_elapsed(end, begin);
        std::cout << message << " : TOTAL TIME ALIVE : " << time_spent << std::endl;
    }

protected:
    struct timespec begin, end;
    std::string message;

    double time_elapsed(struct timespec &now, struct timespec &before) const {
        double time_spent = (double)(now.tv_sec-before.tv_sec);
        time_spent += (now.tv_nsec - before.tv_nsec) / 1000000000.0;
        return time_spent;
    }

private:
};

class ExecTimerSteps : public ExecTimer{
public:
    ExecTimerSteps(char const* message) : ExecTimer(message){
        last = begin;
    }

    double next(char const* step_name){
        double time_spent = step_time();
        std::cout << message << " : Step : " << step_name << " : Time Spent : " << time_spent << std::endl;
        return time_spent;
    }

protected:
    struct timespec now, last;

private:
    double step_time(){
        clock_gettime(CLOCK_MONOTONIC, &now);
        double time_spent = time_elapsed(now, last);
        last = now;
        return time_spent;
    }
};

class ExecTimerCumulative : public ExecTimerSteps{
public:
    ExecTimerCumulative(char const* message) : ExecTimerSteps(message) {
    }

    void lap() {
        laps++;
        std::string lap_name = "Lap " + std::to_string(laps);
        total += next(lap_name.c_str());
        std::cout << "Average time per step: " << total / (double) laps << std::endl;
    }

private:
    double total = 0;
    int laps = 0;
};

#endif //FERRO3D_EXECTIMER_HPP
