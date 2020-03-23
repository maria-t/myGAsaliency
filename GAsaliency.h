#ifndef _GASALIENCY_H_
#define _GASALIENCY_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;


class VTsaliency {

public:
	VTsaliency(){
		//cout << "Object is being created" << endl;
	};

	// vectors to hold center and surround gaussian pyramids
	vector<vector<Mat> > pyr_center_I, pyr_center_RG, pyr_center_BY; 
	vector<vector<Mat> > pyr_surround_I, pyr_surround_RG, pyr_surround_BY;

	// vectors to hold center surround differences
	vector<Mat> on_off_I, off_on_I;
	vector<Mat> on_off_RG, off_on_RG;
	vector<Mat> on_off_BY, off_on_BY;

	Mat salmap;

	vector<Mat> prepare_input(const Mat& v_img);
	// build a multi scale representation based on [Lowe2004]
	vector<vector<Mat> > build_multiscale_pyr(Mat& img, float sigma = 1.f);
	void pyramid_codi(const Mat& v_img);

	void center_surround_diff();

	// computes the final saliency map given that process() was called
  	Mat get_salmap(double &decoded_w1, double &decoded_w2, double &decoded_w3);
  	Mat fuse(vector<Mat> mat_array1);
	Mat last_fuse(vector<Mat> mat_array2, double &idecoded_w1, double &decoded_w2, double &decoded_w3);  

	~VTsaliency(){
		//cout << "Object is being deleted" << endl;
	}

};


//Build multiscale pyramid
vector<vector<Mat> > VTsaliency::build_multiscale_pyr(Mat& mat, float sigma){

	// maximum layer = how often can the image by halfed in the smaller dimension
	// a 320x256 can produce at most 8 layers because 2^8=256

	int stop_layer = 2;
	int max_octaves = min((int)log2(min(mat.rows, mat.cols)), stop_layer)+1;
	// min(8, 4) + 1 = 4 +1 = 5
	Mat tmp = mat.clone();
	
	// reserve space
	vector<vector<Mat> > pyr;
	int start_layer = 0;
	int n_scales = 2;
	pyr.resize(max_octaves-start_layer);
	// start_layer = 0, max_octaves = 5, pyr_size = 5x5
	// compute pyramid as it is done in [Lowe2004]
	float sig_prev = 0.f, sig_total = 0.f;
	
	for(int o = 0; o < max_octaves-start_layer; o++){
		// o = 0, 1, 2, 3, 4
		pyr[o].resize(n_scales+1);
		// n_scales = 2, pyr[0]_size - 3x3 
		// compute an additional scale that is used as the first scale of the next octave
		for(int s = 0; s <= n_scales; s++){
			// s = 0, 1, 2
			Mat& dst = pyr[o][s];
			// 
			// if first scale of first used octave => just smooth tmp
			if(o == 0 && s == 0){
				Mat& src = tmp;

				sig_total = pow(2.0, ((double)s/(double)n_scales))*sigma;
				GaussianBlur(src, dst, Size(), sig_total, sig_total, BORDER_REPLICATE);
				sig_prev = sig_total;
			}

			// if first scale of any other octave => subsample additional scale of previous layer
			else if(o != 0 && s == 0){
				Mat& src = pyr[o-1][n_scales];					
				resize(src, dst, Size(src.cols/2, src.rows/2), 0, 0, INTER_NEAREST);
				sig_prev = sigma;
			}

			// else => smooth an intermediate step
			else{
				sig_total = pow(2.0, ((double)s/(double)n_scales))*sigma;
				float sig_diff = sqrt(sig_total*sig_total - sig_prev*sig_prev);

				Mat& src = pyr[o][s-1];
				GaussianBlur(src, dst, Size(), sig_diff, sig_diff, BORDER_REPLICATE);
				sig_prev = sig_total;
			}
		}
	}

	// erase all the additional scale of each layer
	for(auto& o : pyr){
		o.erase(o.begin()+n_scales);
	}

	return pyr;
}

void VTsaliency::pyramid_codi(const Mat& v_img){

	int n_scales = 2;
	// prepare input image (convert colorspace + split planes)
	vector<Mat> planes = prepare_input(v_img);
	
	// create base pyramids
	vector<vector<Mat> > pyr_base_I, pyr_base_RG, pyr_base_BY;
#pragma omp parallel sections
	{
#pragma omp section
	pyr_base_I = build_multiscale_pyr(planes[0], 1.f);
#pragma omp section
	pyr_base_RG = build_multiscale_pyr(planes[1], 1.f);
#pragma omp section
	pyr_base_BY = build_multiscale_pyr(planes[2], 1.f);
	}

	// recompute sigmas that are needed to reach the desired
	// smoothing for center and surround
	int center_sigma = 3;
	int surround_sigma = 13;
	float adapted_center_sigma = sqrt(pow(center_sigma,2)-1);
	float adapted_surround_sigma = sqrt(pow(surround_sigma,2)-1);

	// reserve space
	pyr_center_I.resize(pyr_base_I.size());
	pyr_center_RG.resize(pyr_base_RG.size());
	pyr_center_BY.resize(pyr_base_BY.size());
	pyr_surround_I.resize(pyr_base_I.size());
	pyr_surround_RG.resize(pyr_base_RG.size());
	pyr_surround_BY.resize(pyr_base_BY.size());

	// for every layer of the pyramid
	for(int o = 0; o < (int)pyr_base_I.size(); o++){
		pyr_center_I[o].resize(n_scales);
		pyr_center_RG[o].resize(n_scales);
		pyr_center_BY[o].resize(n_scales);

		pyr_surround_I[o].resize(n_scales);
		pyr_surround_RG[o].resize(n_scales);
		pyr_surround_BY[o].resize(n_scales);
	


		// for all scales build the center and surround pyramids independently
#pragma omp parallel for
		for(int s = 0; s < n_scales; s++){

			float scaled_center_sigma = adapted_center_sigma*pow(2.0, (double)s/(double)n_scales);
			float scaled_surround_sigma = adapted_surround_sigma*pow(2.0, (double)s/(double)n_scales);

			GaussianBlur(pyr_base_I[o][s], pyr_center_I[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_I[o][s], pyr_surround_I[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

			GaussianBlur(pyr_base_RG[o][s], pyr_center_RG[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_RG[o][s], pyr_surround_RG[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

			GaussianBlur(pyr_base_BY[o][s], pyr_center_BY[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_BY[o][s], pyr_surround_BY[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

		}
	}
}

void VTsaliency::center_surround_diff(){
	int n_scales = 2;
	int on_off_size = pyr_center_I.size()*n_scales;

	on_off_I.resize(on_off_size); off_on_I.resize(on_off_size);
	on_off_RG.resize(on_off_size); off_on_RG.resize(on_off_size);
	on_off_BY.resize(on_off_size); off_on_BY.resize(on_off_size);

	// compute DoG by subtracting layers of two pyramids
	for(int o = 0; o < (int)pyr_center_I.size(); o++){
	#pragma omp parallel for
		for(int s = 0; s < n_scales; s++){
			Mat diff;
			int pos = o*n_scales+s;

			// ========== I channel ==========
			diff = pyr_center_I[o][s]-pyr_surround_I[o][s];
			threshold(diff, on_off_I[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_I[pos], 0, 1, THRESH_TOZERO);

			// ========== RG channel ==========
			diff = pyr_center_RG[o][s]-pyr_surround_RG[o][s];
			threshold(diff, on_off_RG[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_RG[pos], 0, 1, THRESH_TOZERO);

			// ========== BY channel ==========
			diff = pyr_center_BY[o][s]-pyr_surround_BY[o][s];
			threshold(diff, on_off_BY[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_BY[pos], 0, 1, THRESH_TOZERO);
		}
	}
}

#endif
