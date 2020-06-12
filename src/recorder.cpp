#include "recorder.h"

recorder::recorder(std::string exp_name){
	_exp_name = exp_name;
}

std::string ZeroPadNumber(int num)
{
	std::stringstream ss;

	// the number is converted to string with the help of stringstream
	ss << num;
	std::string ret;
	ss >> ret;

	// Append zero chars
	int str_length = ret.length();
	for (int i = 0; i < 7 - str_length; i++)
		ret = "0" + ret;
	return ret;
}

std::string recorder::make_name(std::string ext){
	std::string name;
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, " %Y_%j_%H_%M_%S", timeinfo);
	std::string time_str(buffer);
	name = DIR_REC + _exp_name + ZeroPadNumber(i) + ext;
	i++;
	return name;
}

bool already_exists(std::string file_name){
	std::ifstream attempt(file_name);
	return attempt.good();
}

std::ofstream recorder::setup_record_csv(){
	std::string file_name = make_name(".csv");
	int t_dup = 0;

	while (already_exists(file_name)){
		t_dup++;
		file_name = make_name("_" + std::to_string(t_dup) + ".csv");
	}

	std::ofstream trial(file_name);

	return trial;
}

std::ofstream recorder::setup_record_image(){
	std::string file_name = make_name(".pgm");
	int t_dup = 0;

	while (already_exists(file_name)){
		t_dup++;
		file_name = make_name("_" + std::to_string(t_dup) + ".pgm");
	}

	std::ofstream trial(file_name);

	return trial;
}

std::ofstream recorder::setup_record_csv(std::string custom_name){
	std::string file_name = DIR_REC + custom_name + ".csv";
	int t_dup = 0;

	while (already_exists(file_name)){
		t_dup++;
		file_name = DIR_REC + custom_name + "_" + std::to_string(t_dup) + ".csv";
	}

	std::ofstream trial(file_name);

	return trial;
}

void recorder::txt_record(std::string file_name){
	std::ifstream infile(file_name, std::ifstream::binary);
	std::ofstream outfile(make_name(".txt"), std::ofstream::binary);

	// get size of file
	infile.seekg(0, infile.end);
	long size = infile.tellg();
	infile.seekg(0);

	// allocate memory for file content
	char* buffer = new char[size];

	// read content of infile
	infile.read(buffer, size);

	// write to outfile
	outfile.write(buffer, size);

	// release dynamically-allocated memory
	delete[] buffer;

	outfile.close();
	infile.close();
}
