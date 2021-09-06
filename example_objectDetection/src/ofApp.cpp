#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    model.load("model");


    img_dog.load(ofToDataPath("dog.jpg"));

    vector<string> inputs{"serving_default_input_tensor:0"};
    vector<string> outputs{"StatefulPartitionedCall:0",
                            "StatefulPartitionedCall:1",
                            "StatefulPartitionedCall:2",
                            "StatefulPartitionedCall:3"};

    model.setup(inputs,outputs);
    
    //std::string path(ofToDataPath("dog.jpg"));
    //auto input = cppflow::decode_jpeg(cppflow::read_file(path));

    auto input = ofxTF2::pixelsToTensor(img_dog.getPixels());
    
    input = cppflow::expand_dims(input, 0);
    auto output = model.runMultiModel({input});

    cout << output.size() << endl;

	auto detection_boxes = output[0];
	auto detection_classes = output[1];
    auto detection_scores = output[2];
    auto num_detections = output[3];

    
    ofxTF2::tensorToVector(detection_boxes, detection_boxes_vector);
    ofxTF2::tensorToVector(detection_classes, detection_classes_vector);
    ofxTF2::tensorToVector(detection_scores, detection_scores_vector);
    ofxTF2::tensorToVector(num_detections, num_detections_vector);

    cout << "detection_boxes " << detection_boxes << endl;
    cout << "detection_classes " << detection_classes << endl;
    cout << "detection_scores " << detection_scores << endl;
    cout << "num_detections " << num_detections << endl;

}

//--------------------------------------------------------------
void ofApp::update(){
	// start & stop the model

    
}

//--------------------------------------------------------------
void ofApp::draw(){
    img_dog.draw(0,0);

    ofPushStyle();
    ofSetColor(ofColor::red);
    for(int i = 0; i < detection_scores_vector.size(); i++){

        ofNoFill();

        if(detection_scores_vector[i]>0.5){
            
            int py = detection_boxes_vector[i*4] * img_dog.getHeight();
            int px = detection_boxes_vector[i*4+1] * img_dog.getWidth();
            int pheight = detection_boxes_vector[i*4+2] * img_dog.getHeight() - py;
            int pwidth = detection_boxes_vector[i*4+3] * img_dog.getWidth() - px;
        
            ofDrawRectangle(px, py, pwidth, pheight);
        }
    }
    ofPopStyle();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
