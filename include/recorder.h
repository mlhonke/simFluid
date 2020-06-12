/*
recorder.h Responsible for making record keeping of simulations convenient.

Author: Michael Honke
*/

#ifndef _RECORDER_H_
#define _RECORDER_H_
#define DIR_REC "/home/graphics/Dev/Ferro/Output/"

#include "sim_params.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <cmath>

class recorder{
public:
	std::ofstream setup_record_csv();
	std::ofstream setup_record_image();
	std::ofstream setup_record_csv(std::string custom_name);
	void txt_record(std::string file_name);
	template<typename T> void save_csv(T**data, int res_x, int res_y);
	template<typename T> void save_csv_complex(T**data, char mode, int res_x, int res_y);
	template<typename T> void save_image(T &data, int res_x, int res_y);
	template<typename T> void save_image_complex(T**data, char mode, int res_x, int res_y);
	recorder(std::string exp_name);
private:
	std::string _name;
	std::string _exp_name;
	std::string make_name(std::string ext);
	int i = 0;
};

template <typename T> void recorder::save_csv(T **data, int res_x, int res_y){
	std::ofstream output = recorder::setup_record_csv();

	for (int i = 0; i < res_x; i++){
		for (int j = 0; j < res_y; j++){
			output << data[i][j] << ",";
		}
		output << std::endl;
	}
}

template <typename T> void recorder::save_csv_complex(T **data, char mode, int res_x, int res_y){
	std::ofstream output = recorder::setup_record_csv();

	if (mode == 'r'){
		for (int i = 0; i < res_x; i++){
			for (int j = 0; j < res_y; j++){
				output << data[i][j].x << ",";
			}
			output << std::endl;
		}
	} else if (mode == 'i') {
		for (int i = 0; i < res_x; i++){
			for (int j = 0; j < res_y; j++){
				output << data[i][j].y << ",";
			}
			output << std::endl;
		}
	} else {
		for (int i = 0; i < res_x; i++){
			for (int j = 0; j < res_y; j++){
				output << std::sqrt( std::pow(data[i][j].x, 2) + std::pow(data[i][j].y, 2)) << ",";
			}
			output << std::endl;
		}
	}
}

template <typename T> void recorder::save_image(T& data, int res_x, int res_y){
	std::ofstream output = recorder::setup_record_image();

	output << "P2" << std::endl;
	output << res_x << " " << res_y << std::endl;

	scalar_t max = 0;
	max = data.array().abs().maxCoeff();

	output << (int) 10240 << std::endl;
	scalar_t scale = max / 10240;

	for (int i = 0; i < res_x; i++){
		for (int j = 0; j < res_y; j++){
			output << (int) (std::abs(data(i, j)) / scale) << " ";
		}
		output << std::endl;
	}
}

template <typename T> void recorder::save_image_complex(T **data, char mode, int res_x, int res_y){
	std::ofstream output = recorder::setup_record_image();

	output << "P2" << std::endl;
	output << res_x << " " << res_y << std::endl;

	scalar_t max = 0;
	if (mode == 'r' || mode == 'i'){
		for (int i = 0; i < res_x; i++){
			for (int j = 0; j < res_y; j++){
				if (data[i][j].x > max)
					max = data[i][j].x;
				if (data[i][j].y > max)
					max = data[i][j].y;
			}
		}
	} else {
		for (int i = 0; i < res_x; i++){
			for (int j = 0; j < res_y; j++){
				if (std::sqrt( std::pow(data[i][j].x, 2) + std::pow(data[i][j].y, 2)) > max)
					max = std::sqrt( std::pow(data[i][j].x, 2) + std::pow(data[i][j].y, 2));
			}
		}
	}

	output << (int) 10240 << std::endl;
	scalar_t scale = max / 10240;

	if (mode == 'r'){
		for (int i = 0; i < res_x; i++){
			for (int j = 0; j < res_y; j++){
				output << (int) (data[i][j].x / scale) << " ";
			}
			output << std::endl;
		}
	} else if (mode == 'i') {
		for (int i = 0; i < res_x; i++){
			for (int j = 0; j < res_y; j++){
				output << (int) (data[i][j].y / scale) << " ";
			}
			output << std::endl;
		}
	} else {
		for (int i = 0; i < res_x; i++){
			for (int j = 0; j < res_y; j++){
				output << (int) (std::sqrt( std::pow(data[i][j].x, 2) + std::pow(data[i][j].y, 2)) / scale) << " ";
			}
			output << std::endl;
		}
	}
}

#endif
