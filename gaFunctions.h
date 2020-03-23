#ifndef _GA_H_
#define _GA_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono> 
#include <random>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <functional>
#include <ctime>
#include <iomanip>
#include <string>
#include <fstream>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xoperation.hpp"
#include "GAsaliency.h"

using namespace xt::placeholders;

Mat read_image(string const &img_name);
void show_image(const Mat &img, const char* window_name);
Mat read_human_segImg(string const &img_name);

int threshold_value = 120;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;
Mat sal_map, thresholded_map, map_copy;
Mat new_map, new_map_copy;
Mat new_binarySaliency;
const char* window_name = "Threshold Map";
const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

struct FitnessFunction
{
    double DecodedW1;
    double DecodedW2;
    double DecodedW3;
    double FitnessValue;
}; 

// static FitnessFunction Fitness(const xt::xarray<int> &chromosome);
static FitnessFunction Fitness(const xt::xarray<int> &chromosome)
{
    int lb_w1 = 0, ub_w1 = 1;      
    int len_w1 = chromosome.size()/3; 
    int lb_w2 = 0, ub_w2 = 1;
    int len_w2 = chromosome.size()/3;
    int lb_w3 = 0, ub_w3 = 1;
    int len_w3 = chromosome.size()/3;
  
    double precision_w1 = (ub_w1 - lb_w1)/(pow(2, len_w1)-1);
    double precision_w2 = (ub_w2 - lb_w2)/(pow(2, len_w2)-1);
    double precision_w3 = (ub_w3 - lb_w3)/(pow(2, len_w3)-1);

    xt::xarray<int> w1 = xt::view(chromosome, xt::range(0, chromosome.size()/3));
    //std::cout << "w1 variable is: " << w1 << std::endl;  
    xt::xarray<int> w2 = xt::view(chromosome, xt::range(chromosome.size()/3, 2*chromosome.size()/3));
    //std::cout << "w2 variable is: " << w2 << std::endl;
    xt::xarray<int> w3 = xt::view(chromosome, xt::range(2*chromosome.size()/3, chromosome.size()));
    //std::cout << "w3 variable is: " << w3 << std::endl;

    // sum(bit*(2^z)) for w1
    int z = 0;
    int w1_bit_sum = 0;
    int w1_bit; 
    for (auto it = w1.rbegin(); it != w1.rend(); ++it)
    {   
        // std::cout << std::endl;
        //cout << "index is: " << it << endl;
        // std::cout << "bit is: " << *it << std::endl;
        w1_bit = (*it) * pow(2, z); 
        // std::cout << *it << " * " << "(2^" << z << ") = " << w1_bit << std::endl;
        w1_bit_sum += w1_bit; 
        z += 1; 
    }

    // std::cout << std::endl;
    // std::cout << "sum(bit*(2^z)) is : " << w1_bit_sum << std::endl;
    
    // sum(bit*(2^z)) for w2
    z = 0;
    int w2_bit_sum = 0;
    int w2_bit; 
    for (auto it = w2.rbegin(); it != w2.rend(); ++it)
    {   
        // std::cout << std::endl;
        //cout << "index is: " << it << endl;
        // std::cout << "bit is: " << *it << std::endl;
        w2_bit = (*it) * pow(2, z); 
        // std::cout << *it << " * " << "(2^" << z << ") = " << w2_bit << std::endl;
        w2_bit_sum += w2_bit; 
        z += 1; 
    }

    // std::cout << std::endl;
    // std::cout << "sum(bit*(2^z)) is : " << w2_bit_sum << std::endl;

    // sum(bit*(2^z)) for w3
    z = 0;
    int w3_bit_sum = 0;
    int w3_bit; 
    for (auto it = w3.rbegin(); it != w3.rend(); ++it)
    {   
        // std::cout << std::endl;
        //cout << "index is: " << it << endl;
        // std::cout << "bit is: " << *it << std::endl;
        w3_bit = (*it) * pow(2, z); 
        // std::cout << *it << " * " << "(2^" << z << ") = " << w3_bit << std::endl;
        w3_bit_sum += w3_bit; 
        z += 1; 
    }

    // std::cout << std::endl;
    // std::cout << "sum(bit*(2^z)) is : " << w3_bit_sum << std::endl;
    
    double decoded_w1 = w1_bit_sum * precision_w1 + lb_w1;
    double decoded_w2 = w2_bit_sum * precision_w2 + lb_w2;
    double decoded_w3 = w3_bit_sum * precision_w3 + lb_w3;
    // double obj_func_val = pow(pow(decoded_w2, 2) + decoded_w1 - 11, 2) + pow(decoded_w2 + pow(decoded_w1, 2) - 7, 2);
    
    // read image
	string input_fname = "/home/maria/TestingMySaliency/GA/111876273311/src_color/111876273311.png";
    Mat color_image = read_image(input_fname);
    show_image(color_image, "Color Image");

	// read groundtruth image
	string groundtruth_fname = "/home/maria/TestingMySaliency/GA/111876273311/human_seg/111876273311_1.png";
	Mat human_seg = read_human_segImg(groundtruth_fname);
	human_seg.convertTo(human_seg, CV_32F);
	show_image(human_seg, "Groundtruth");
	//imwrite("/home/maria/TestingMySaliency/GA/fishing_on_water/groundtruth.png", human_seg);
	// waitKey(0);

    // generate initial saliency map
    VTsaliency myMap;
    myMap.pyramid_codi(color_image);
    myMap.center_surround_diff();
	sal_map = myMap.get_salmap(decoded_w1, decoded_w1, decoded_w1);
	show_image(sal_map, "Saliency Map");

    //evaluate threshold for binary saliency map
	// namedWindow(window_name, WINDOW_AUTOSIZE); // Create a window to display results
	// createTrackbar(trackbar_type,
	//                 window_name, &threshold_type,
	//                 max_type, threshold_map); // Create a Trackbar to choose type of Threshold
	// createTrackbar(trackbar_value,
	//                 window_name, &threshold_value,
	//                 max_value, threshold_map); // Create a Trackbar to choose Threshold value
	// threshold_map(0, 0); // Call the function to initialize
	// waitKey();

	// // save initial saliency map
	// // sal_map *= 255.f;
	// // imwrite("/home/maria/TestingMySaliency/GA/111876273311/saliency.png", map);

    // copy of saliency map
	sal_map.copyTo(map_copy);
	show_image(map_copy, "Saliency Copy Map");

	// generate binarized Saliency Map
	map_copy *= 255.f;
	Mat binarySaliency;
	threshold(map_copy, binarySaliency, threshold_value, max_binary_value, threshold_type);
	show_image(binarySaliency, "Binary Saliency Map");
	// waitKey();
    //imwrite("/home/maria/TestingMySaliency/GA/111876273311/binary_saliency.png", binarySaliency);

    // fitness function	
	double euclidean_distance = norm(binarySaliency, human_seg, NORM_L2);

    return {decoded_w1, decoded_w2, decoded_w3, euclidean_distance};
}


xt::xarray<int> SelectParent(xt::xarray<int> &all_solutions)
{
    // unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    // std::default_random_engine engine(seed);
    
    // get 3 random integers (k=3 for tournament selection)
    // xt::xarray<int> indices_list = xt::random::randint<int>({3}, 0, 20, engine);
    xt::xarray<int> ind_list = xt::arange<int>(0, all_solutions.shape(0), 1);
    auto indices_list = xt::random::choice(ind_list, 3, false);
    // std::cout << "Selected possible parents out of population : " << indices_list << std::endl;  

    // get the 3 possible parents for selection from "all solutions"
    auto possible_parent_1 = xt::view(all_solutions, indices_list(0), xt::all());
    auto possible_parent_2 = xt::view(all_solutions, indices_list(1), xt::all());
    auto possible_parent_3 = xt::view(all_solutions, indices_list(2), xt::all());
    
    // get objective value (fitness) for each possible parent
    auto fitness_parent_1 = Fitness(possible_parent_1).FitnessValue;
    auto fitness_parent_2 = Fitness(possible_parent_2).FitnessValue;
    auto fitness_parent_3 = Fitness(possible_parent_3).FitnessValue;

    // auto fitness_parent_1 = Fitness(possible_parent_1, binarySaliency, human_seg).FitnessValue;
    // auto fitness_parent_2 = Fitness(possible_parent_2, binarySaliency, human_seg).FitnessValue;
    // auto fitness_parent_3 = Fitness(possible_parent_3, binarySaliency, human_seg).FitnessValue;
    // std::cout << "fv_1 = " << fitness_parent_1 << std::endl;
    // std::cout << "fv_2 = " << fitness_parent_2 << std::endl;
    // std::cout << "fv_3 = " << fitness_parent_3 << std::endl;

    // which parent is the best
    auto min_fitness = std::min({fitness_parent_1, fitness_parent_2, fitness_parent_3});
    //cout << min_fitness << endl;
    
    xt::xarray<int> selected_parent;
    if (min_fitness == fitness_parent_1){
        selected_parent = possible_parent_1;   
    }else if (min_fitness == fitness_parent_2){
        selected_parent = possible_parent_2;
    }else {
        selected_parent = possible_parent_3;
    }

    return selected_parent;
}

xt::xarray<int> Crossover(xt::xarray<int> &selected_parent_1, xt::xarray<int> &selected_parent_2, double &prob_xover)
{
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);

    xt::xarray<int> children;
    auto xover_or_not = xt::random::rand<double>({1,1}, 0, 1, engine);
    // std::cout << "Do we crossover or not? --> " << xover_or_not(0) << std::endl;

    if (xover_or_not(0) < prob_xover){
        
        xt::xarray<int> xover_index_1 = xt::random::randint<int>({1, 1}, 0, selected_parent_1.size(), engine);
        xt::xarray<int> xover_index_2 = xt::random::randint<int>({1, 1}, 0, selected_parent_2.size(), engine);

        // std::cout << xover_index_1(0) << std::endl;
        // std::cout << xover_index_2(0) << std::endl;    
    
        // ensuring the crossover points are not the same
        while (xover_index_1(0) == xover_index_2(0)){
            xover_index_2 = xt::random::randint<int>({1, 1}, 0, selected_parent_2.size(), engine);    
        }

        // std::cout << "First Crossover Point --> " << xover_index_1(0) << std::endl;
        // std::cout << "Second Crossover Point --> " << xover_index_2(0) << std::endl;    
    
        if (xover_index_1(0) < xover_index_2(0)){

            // slicing parent 1
            auto first_seg_selected_parent_1 = xt::view(selected_parent_1, xt::range(_, xover_index_1(0)), xt::all());
            auto mid_seg_selected_parent_1 = xt::view(selected_parent_1, xt::range(xover_index_1(0), xover_index_2(0) + 1), xt::all());
            auto last_seg_selected_parent_1 = xt::view(selected_parent_1, xt::range(xover_index_2(0) + 1, _), xt::all());
            // std::cout << "1st Segment of Parent 1 for crossover --> " << first_seg_selected_parent_1 << std::endl;
            // std::cout << "2nd Segment of Parent 1 for crossover --> " << mid_seg_selected_parent_1 << std::endl;
            // std::cout << "3rd Segment of Parent 1 for crossover --> " << last_seg_selected_parent_1 << std::endl;

            // slicing parent 2
            auto first_seg_selected_parent_2 = xt::view(selected_parent_2, xt::range(_, xover_index_1(0)), xt::all());
            auto mid_seg_selected_parent_2 = xt::view(selected_parent_2, xt::range(xover_index_1(0), xover_index_2(0) + 1), xt::all());
            auto last_seg_selected_parent_2 = xt::view(selected_parent_2, xt::range(xover_index_2(0) + 1, _), xt::all());
            // std::cout << "1st Segment of Parent 2 for crossover --> " << first_seg_selected_parent_2 << std::endl;
            // std::cout << "2nd Segment of Parent 2 for crossover --> " << mid_seg_selected_parent_2 << std::endl;
            // std::cout << "3rd Segment of Parent 2 for crossover --> " << last_seg_selected_parent_2 << std::endl;

            // creating children
            xt::xarray<int> child_1 = xt::concatenate(xt::xtuple(first_seg_selected_parent_1, mid_seg_selected_parent_2, last_seg_selected_parent_1));
            xt::xarray<int> child_2 = xt::concatenate(xt::xtuple(first_seg_selected_parent_2, mid_seg_selected_parent_1, last_seg_selected_parent_2));
            // std::cout << "Child 1 : " << child_1 << std::endl;
            // std::cout << "Child 2 : " << child_2 << std::endl;

            children = xt::concatenate(xt::xtuple(child_1, child_2));

        }else {

            // slicing parent 1
            auto first_seg_selected_parent_1 = xt::view(selected_parent_1, xt::range(_, xover_index_2(0)), xt::all());
            auto mid_seg_selected_parent_1 = xt::view(selected_parent_1, xt::range(xover_index_2(0), xover_index_1(0) + 1), xt::all());
            auto last_seg_selected_parent_1 = xt::view(selected_parent_1, xt::range(xover_index_1(0) + 1, _), xt::all());
            // std::cout << "1st Segment of Parent 1 for crossover --> " << first_seg_selected_parent_1 << std::endl;
            // std::cout << "2nd Segment of Parent 1 for crossover --> " << mid_seg_selected_parent_1 << std::endl;
            // std::cout << "3rd Segment of Parent 1 for crossover --> " << last_seg_selected_parent_1 << std::endl;

            // slicing parent 2
            auto first_seg_selected_parent_2 = xt::view(selected_parent_2, xt::range(_, xover_index_2(0)), xt::all());
            auto mid_seg_selected_parent_2 = xt::view(selected_parent_2, xt::range(xover_index_2(0), xover_index_1(0) + 1), xt::all());
            auto last_seg_selected_parent_2 = xt::view(selected_parent_2, xt::range(xover_index_1(0) + 1, _), xt::all());
            // std::cout << "1st Segment of Parent 2 for crossover --> " << first_seg_selected_parent_2 << std::endl;
            // std::cout << "2nd Segment of Parent 2 for crossover --> " << mid_seg_selected_parent_2 << std::endl;
            // std::cout << "3rd Segment of Parent 2 for crossover --> " << last_seg_selected_parent_2 << std::endl;

            // creating children
            xt::xarray<int> child_1 = xt::concatenate(xt::xtuple(first_seg_selected_parent_1, mid_seg_selected_parent_2, last_seg_selected_parent_1));
            xt::xarray<int> child_2 = xt::concatenate(xt::xtuple(first_seg_selected_parent_2, mid_seg_selected_parent_1, last_seg_selected_parent_2));
            // std::cout << "Child 1 : " << child_1 << std::endl;
            // std::cout << "Child 2 : " << child_2 << std::endl;

            children = xt::concatenate(xt::xtuple(child_1, child_2));

        }
    
    }else {
        auto child_1 = selected_parent_1;
        auto child_2 = selected_parent_2;

        // std::cout << "Child 1 : " << child_1 << std::endl;
        // std::cout << "Child 2 : " << child_2 << std::endl;

        children = xt::concatenate(xt::xtuple(child_1, child_2));
    }

    return children;
}

xt::xarray<int> Mutation(xt::xarray<int> &child_1, xt::xarray<int> &child_2, double &prob_mutation){

    xt::xarray<int> mutated_children;
    xt::xarray<int> mutated_child_1, mutated_child_2;

    // generate mutated child 1 ---- flip-bit mutation
    int mutated_index = 0; 
    for (int i = 0; i < child_1.size(); i++){

        unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
        std::default_random_engine engine(seed);

        xt::xarray<double> mutate_or_not = xt::random::rand<double>({1,1}, 0, 1, engine);
        // std::cout << "Do we mutate or not? --> " << mutate_or_not(0) << std::endl;
        
        if (mutate_or_not(0) < prob_mutation){
            // std::cout << "YES, we will mutate gene in position : " << mutated_index << std::endl;

            if (child_1(mutated_index) == 0){
                child_1(mutated_index) = 1;
            }else {
                child_1(mutated_index) = 0;
            }

            mutated_child_1 = child_1;

            mutated_index = mutated_index + 1;
            
            // std::cout << "New child_1 is : " << child_1 << std::endl;


        }else {
            // std::cout << "NO, we will not mutate gene in position : " << mutated_index << std::endl;
            mutated_child_1 = child_1;

            mutated_index = mutated_index + 1;
        }
    }

    // generate mutated child 2 ---- flip-bit mutation
    mutated_index = 0; 
    for (int i = 0; i < child_2.size(); i++){

        unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
        std::default_random_engine engine(seed);

        xt::xarray<double> mutate_or_not = xt::random::rand<double>({1,1}, 0, 1, engine);
        // std::cout << "Do we mutate or not? --> " << mutate_or_not(0) << std::endl;
        
        if (mutate_or_not(0) < prob_mutation){
            // std::cout << "YES, we will mutate gene in position : " << mutated_index << std::endl;

            if (child_2(mutated_index) == 0){
                child_2(mutated_index) = 1;
            }else {
                child_2(mutated_index) = 0;
            }

            mutated_child_2 = child_2;

            mutated_index = mutated_index + 1;
            
            // std::cout << "New child_2 is : " << child_2 << std::endl;


        }else {
            // std::cout << "NO, we will not mutate gene in position : " << mutated_index << std::endl;
            mutated_child_2 = child_2;

            mutated_index = mutated_index + 1;
        }
    }
    
    mutated_children = xt::concatenate(xt::xtuple(mutated_child_1, mutated_child_2));

    return mutated_children;
}

// read image
Mat read_image(string const &img_name){
	Mat img;
	img = imread(img_name, IMREAD_COLOR);
	if( img.empty() ){
		cout << "Could not open or find the image!\n" << endl;
	}
	return img;
}

// show image
void show_image(const Mat &img, const char* window_name){
	namedWindow(window_name);
	imshow(window_name, img);
	// waitKey(); 
}

// read human segmented image - Groundtruth
Mat read_human_segImg(string const &img_name){
	Mat segImg;
	segImg = imread(img_name, IMREAD_COLOR);
	if(segImg.empty() ){
		cout << "Could not open or find the image!\n" << endl;
	}
	
	Mat bgr[3];   //destination array
	split(segImg,bgr); //split source  

	Mat mask1, mask2;
	threshold(bgr[0], mask1, 120, 255, 0); //120
	threshold(bgr[2], mask2, 120, 255, 0);
	Mat mask; 
	bitwise_or(mask1, mask2, mask);

	return mask;
}

// thresholed saliency map
static void threshold_map(int, void*){
    /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
    */
    threshold(sal_map, thresholded_map, threshold_value, max_binary_value, threshold_type);
    //imshow(window_name, thresholded_map);
}

vector<Mat> VTsaliency::prepare_input(const Mat& v_img){

	CV_Assert(v_img.channels() == 3);
	vector<Mat> planes;
	planes.resize(3);

	Mat v_converted;
	// convert to float
	v_img.convertTo(v_converted, CV_32FC3);

	// scale down to range [0:1]
	vector<Mat> planes_bgr;
	split(v_converted, planes_bgr);
	
	planes[0] = planes_bgr[0] + planes_bgr[1] + planes_bgr[2];
	planes[0] /= 3*255.f;

	planes[1] = planes_bgr[2] - planes_bgr[1];
	planes[1] /= 255.f;

	planes[2] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f;
	planes[2] /= 255.f;
		
	return planes;

}

Mat VTsaliency::get_salmap(double &decoded_w1, double &decoded_w2, double &decoded_w3){

	// INTENSITY feature maps
	vector<Mat> feature_visible_INTENSITY;
	feature_visible_INTENSITY.push_back(fuse(on_off_I));
	feature_visible_INTENSITY.push_back(fuse(off_on_I));
	
	// RG feature maps
	vector<Mat> feature_visible_RG;
	feature_visible_RG.push_back(fuse(on_off_RG));
	feature_visible_RG.push_back(fuse(off_on_RG));

	// BY feature maps
	vector<Mat> feature_visible_BY;
	feature_visible_BY.push_back(fuse(on_off_BY));
	feature_visible_BY.push_back(fuse(off_on_BY));

	// conspicuity maps
	vector<Mat> conspicuity_maps;
	conspicuity_maps.push_back(fuse(feature_visible_INTENSITY));
	conspicuity_maps.push_back(fuse(feature_visible_RG));
	conspicuity_maps.push_back(fuse(feature_visible_BY));

	// saliency map
	salmap = last_fuse(conspicuity_maps, decoded_w1, decoded_w2, decoded_w3);

	// normalize output to [0,1]
	double mi, ma;
	minMaxLoc(salmap, &mi, &ma);
	salmap = (salmap-mi)/(ma-mi);

	Size size(300, 199);
	// resize to original image size
	resize(salmap, salmap, size, 0, 0, INTER_CUBIC);

	return salmap;
}

//Fuse maps using operation
Mat VTsaliency::fuse(vector<Mat> maps){
	
	// resulting map that is returned
	Mat fused = Mat::zeros(maps[0].size(), CV_32F);
	int n_maps = maps.size();	// no. of maps to fuse
	vector<Mat> resized;		// temp. array to hold the resized maps
	resized.resize(n_maps);		// reserve space (needed to use openmp for parallel resizing)

	// ========== ARITHMETIC MEAN ==========

		for(int i = 0; i < n_maps; i++){
			//cout << "for i = "<< i << " fused size is = " << fused.size() << " and maps size is = " << maps[i].size() << endl;
			if(fused.size() != maps[i].size()){
				resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
			else{
				resized[i] = maps[i];
			}
		}

		// fused = fused + new (weights_vector[i] * resized[i])

		for(int i = 0; i < n_maps; i++){
			add(fused, resized[i], fused, Mat(), CV_32F);
		}	

		fused /= (float)n_maps;

	return fused;
}

//Fuse maps using operation
Mat VTsaliency::last_fuse(vector<Mat> maps, double &decoded_w1, double &decoded_w2, double &decoded_w3){
	
	// resulting map that is returned
	Mat fused = Mat::zeros(maps[0].size(), CV_32F);
	int n_maps = maps.size();	// no. of maps to fuse
	vector<Mat> resized;		// temp. array to hold the resized maps
	resized.resize(n_maps);		// reserve space (needed to use openmp for parallel resizing)

	// ========== ARITHMETIC MEAN ==========

		for(int i = 0; i < n_maps; i++){
			//cout << "for i = "<< i << " fused size is = " << fused.size() << " and maps size is = " << maps[i].size() << endl;
			if(fused.size() != maps[i].size()){
				resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
			else{
				resized[i] = maps[i];
			}
		}

		fused = (decoded_w1 / (decoded_w1 + decoded_w2 + decoded_w3)) * resized[0] + (decoded_w2 / (decoded_w1 + decoded_w2 + decoded_w3)) * resized[1] + (decoded_w3 / (decoded_w1 + decoded_w2 + decoded_w3))* resized[2];

	return fused;
}

#endif // _GA_H_