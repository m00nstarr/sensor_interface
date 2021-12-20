#include <stdio.h>
#include <math.h>
//#include<string>

typedef enum
{
	RELU, SIGMOID
}Activation;

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float relu(float x) {
	return x > 0.0f ? x : 0.0f;
}

//class Tensor {
//public:
//	Tensor() {};
//	Tensor(vector<float> data) {
//		this->data = data;
//		this->shape.push_back(data.size());
//	}
//
//	Tensor(int dim) {
//		this->shape.push_back(dim);
//		this->data = vector<float>(dim);
//	}
//	vector<float> data;
//	vector<int> shape = { 0 };
//
//};

typedef struct _Linear {
	int in_dim, out_dim;
	float** weight;
	float* bias;
	float (*activation)(float);
} Linear;

typedef struct _Model {
	int module_num;
	Linear** modules;

}Model;

void Linear_construct(Linear* linear, int out_dim, int in_dim, float* bias, Activation act) {
	//printf("%d, %d", out_dim, in_dim);
	linear->bias = bias;
	linear->in_dim = in_dim;
	linear->out_dim = out_dim;
	if (act == RELU) {
		linear->activation = relu;
	}
	else if (act == SIGMOID) {
		linear->activation = sigmoid;
	}
}

void Model_construct(Model* model, int module_num) {
	model->module_num = module_num;
	model->modules = (Linear**)malloc(sizeof(Linear*) * module_num);
}

void Model_append(Model* model, Linear* linear, int idx) {
	model->modules[idx - 1] = linear;
}

float* forward(Model* model, float *input) {
	for (int m = 0; m < model->module_num; m++) {
		Linear* linear = model->modules[m];
		float* output = (float*)calloc(linear->out_dim, sizeof(float));
		for (int i = 0; i < linear->out_dim; i++) {
			float d = 0.0;
			for (int j = 0; j < linear->in_dim; j++) {
				d += linear->weight[i][j] * input[j];
			}
			output[i] = linear->activation(d + linear->bias[i]);
		}
		if(m!=0){
			free(input);
		}
		input = output;
	}
	return input;
}

