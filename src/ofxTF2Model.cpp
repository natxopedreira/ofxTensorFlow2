/*
 * ofxTensorFlow2
 *
 * Copyright (c) 2021 ZKM | Hertz-Lab
 * Paul Bethge <bethge@zkm.de>
 * Dan Wilcox <dan.wilcox@zkm.de>
 *
 * BSD Simplified License.
 * For information on usage and redistribution, and for a DISCLAIMER OF ALL
 * WARRANTIES, see the file, "LICENSE.txt," in this distribution.
 *
 * This code has been developed at ZKM | Hertz-Lab as part of „The Intelligent 
 * Museum“ generously funded by the German Federal Cultural Foundation.
 */

#include "ofxTF2Model.h"

#include "ofFileUtils.h"
#include "ofUtils.h"
#include "ofLog.h"


namespace ofxTF2 {

Model::Model(const std::string & modelPath) {
	load(modelPath);
}

Model::~Model(){
	clear();
}


void Model::setup(const ModelSettings & settings) {
	settings_ = settings;
}

bool Model::load(const std::string & modelPath) {
	Model::clear();
	std::string path = ofToDataPath(modelPath);
	if (!ofDirectory::doesDirectoryExist(path)){
		ofLogError() << "Model: model path not found: " << modelPath;
		return false;
	}
	auto model = new cppflow::model(path);
	if (!model){
		modelPath_ = "";
		ofLogError() << "Model: model could not be initialized!";
		return false;
	}	
	model_ = model;
	modelPath_ = modelPath;
	ofLogVerbose() << "Model: loaded model: " << modelPath_;
	return true;
}

void Model::clear() {
	if (model_){
		ofLogVerbose() << "Model: clear model" << modelPath_;
		delete model_;
		model_ = nullptr;
		modelPath_ = "";
	}
}

cppflow::tensor Model::runModel(const cppflow::tensor & input) const {
	if (model_){
		return (*model_)({{settings_.inputName_, input}}, {settings_.outputName_})[0];
	}
	else{
		ofLog() << "Model: no model loaded! Returning tensor containing -1.";
		return cppflow::tensor(-1);
	}
}

bool Model::isLoaded() {
	return model_ != nullptr;
}

}; // end namespace ofxTF2
