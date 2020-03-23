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
#include "gaFunctions.h"

using namespace cv;
using namespace std;
using namespace xt::placeholders;

static FitnessFunction Fitness(const xt::xarray<int> &chromosome);
// static FitnessFunction Fitness(const xt::xarray<int> &chromosome, Mat &binarySaliency, Mat &human_seg);
xt::xarray<int> SelectParent(xt::xarray<int> &all_solutions);
xt::xarray<int> Crossover(xt::xarray<int> &selected_parent_1, xt::xarray<int> &selected_parent_2, double &prob_xover);
xt::xarray<int> Mutation(xt::xarray<int> &child_1, xt::xarray<int> &child_2, double &prob_mutation);


int main()
{

    double prob_xover = 1; // probability of crossover
    double prob_mutation = 0.5; // probability of mutation
    long unsigned int population = 200; // population number
    long unsigned int generations = 180; // generation number

    unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    default_random_engine engine(seed);

    xt::xarray<int> chromosome              // initial solution
    {0, 0, 1, 1, 1,                         // w1 variable
     0, 1, 1, 1, 0,                         // w2 variable
     1, 1, 0, 0, 0};                        // w3 variable     

    // cout << "Chromosome is: " << chromosome << endl;
    // cout << "Chromosome length is: " << chromosome.size() << endl;

    // create initial population
    // shuffle the elements in the initial solution 
    // create a pool of solutions
    xt::random::shuffle(chromosome, engine);
    xt::xarray<int> all_solutions = chromosome;
    
    for (auto i = 0; i < population - 1; i++)
    {
        xt::random::shuffle(chromosome);
        all_solutions = xt::concatenate(xt::xtuple(all_solutions, chromosome)); 
    }
    
    all_solutions.reshape({population, chromosome.size()});
    cout << "My Initial Population is :"  << endl;
    cout << all_solutions << endl;   


    // for storing the best fitness value of each generation
    vector<double> best_of_a_generation;
    xt::xarray<int> indx_solution;

    // for storing the best solutions
    xt::xarray<int>::shape_type sh = {chromosome.size()};
    xt::xarray<int> best_solutions = xt::empty<int>(sh);

    // for best of the best solutions
    xt::xarray<int> indx_best_of_the_best;

    // get starting timepoint 
    auto start = chrono::high_resolution_clock::now(); 
    // start at generation No1
    int generation = 1; 

    // iterate over the generations
    for (auto i = 0; i < generations; i++){
        
        cout << " Generation: #" << generation << endl;
        // for storing the new pool of solutions
        xt::xarray<int>::shape_type sh0 = {2, chromosome.size()};
        xt::xarray<int> new_population = xt::empty<int>(sh0);

        // for storing the fitness values of each generation
        vector<double> fv_per_generation;

        int family = 1; 
        for (auto k = 0; k < (population/2); k++){    

            cout << " Family: #" << family << endl;

            // select 2 parents using tournament selection
            xt::xarray<int> parent_1 = SelectParent(all_solutions);
            xt::xarray<int> parent_2 = SelectParent(all_solutions);

            // crossover the 2 parents to get 2 children
            xt::xarray<int> children = Crossover(parent_1, parent_2, prob_xover);
            children.reshape({2, chromosome.size()});
            xt::xarray<int> child_1 = xt::view(children, 0, xt::all());
            xt::xarray<int> child_2 = xt::view(children, 1, xt::all());

            // mutate the 2 children using flip-bit mutation
            xt::xarray<int> mutated_children = Mutation(child_1, child_2, prob_mutation);
            mutated_children.reshape({2, chromosome.size()});
            xt::xarray<int> mutated_child_1 = xt::view(mutated_children, 0, xt::all());
            xt::xarray<int> mutated_child_2 = xt::view(mutated_children, 1, xt::all());

            // getting the fitness value for the 2 mutated children
            FitnessFunction fit_mutant_1 = Fitness(mutated_child_1);
            double fv_mutant_1 = fit_mutant_1.FitnessValue;
            
            FitnessFunction fit_mutant_2 = Fitness(mutated_child_2);
            double fv_mutant_2 = fit_mutant_2.FitnessValue;

            // keep track of mutants and their fitness values
            cout << "Fitness Value - Mutant Child 1 : " << fv_mutant_1 << " , Generation #" << generation << endl; 
            cout << "Fitness Value - Mutant Child 2 : " << fv_mutant_2 << " , Generation #" << generation << endl; 

            new_population = xt::concatenate(xt::xtuple(new_population, mutated_children));

            fv_per_generation.push_back(fv_mutant_1);
            fv_per_generation.push_back(fv_mutant_2);

            family = family + 1;
        } 
        
        // prepare the new population for the next generation
        new_population = xt::view(new_population, xt::drop(0, 1));
        new_population.reshape({population, chromosome.size()});
        // cout << "Population in Generation # " << generation + 1 << ":" << endl;
        // cout << new_population << endl;
        all_solutions = new_population;
        
        // fitness values of all families for each generation
        vector<size_t> sh1 = {population, 1};
        xt::xarray<double> fit_val_generation = xt::adapt(fv_per_generation, sh1);
        // cout << fit_val_generation << endl;

        // best of a generation
        double best_fv = xt::sort(fit_val_generation, 0)(0,0);
        best_of_a_generation.push_back(best_fv);

        // find position of best solution for each generation
        indx_solution = xt::argmin(fit_val_generation);
        // stack the best solutions
        xt::xarray<int> best_solution = xt::view(all_solutions, indx_solution(0), xt::all());
        best_solutions = xt::concatenate(xt::xtuple(best_solutions, best_solution));

        generation = generation + 1;
    }

    cout << "----------------------------------------------------------------" << endl;

    // get ending timepoint 
    auto stop = chrono::high_resolution_clock::now();

    // the solution the algorithm converged to
    xt::xarray<int> best_solution_convergence = xt::view(all_solutions, indx_solution(0), xt::all());
    cout << "Convergence: " << best_solution_convergence << endl; 

    // print best fitness values across generations
    // for (auto it = best_of_a_generation.cbegin(); it != best_of_a_generation.cend(); it++)
	// {
	// 	cout << *it << ' ';
	// }
    // cout << endl;

    // best solutions across generations
    best_solutions.reshape({generations+1, chromosome.size()});
    best_solutions = xt::view(best_solutions, xt::drop(0));
    // cout << "Best solutions across generations: " << endl;
    // cout << best_solutions << endl;

    // best fitness value out of all generations
    vector<size_t> sh2 = {generations, 1};
    xt::xarray<double> best_of_a_generation_xarray = xt::adapt(best_of_a_generation, sh2);
    // cout << best_of_a_generation_xarray << endl;
    indx_best_of_the_best = xt::argmin(best_of_a_generation_xarray);
    // cout << indx_best_of_the_best << endl;
    xt::xarray<int> best_of_the_best = xt::view(best_solutions, indx_best_of_the_best(0), xt::all());
    cout << "Best of the best solutions: " << best_of_the_best << endl;

    // execution time of algorithm
    auto duration = chrono::duration_cast<chrono::seconds>(stop - start); 
    cout << "Time taken by function: " << duration.count() << " seconds" << endl;

    cout << "----------------------------------------------------------------" << endl;
    // decoded values and fitness value for convergence 
    auto fit_convergence = Fitness(best_solution_convergence);
    auto convergence_decoded_w1 = fit_convergence.DecodedW1;
    auto convergence_decoded_w2 = fit_convergence.DecodedW2;
    auto convergence_decoded_w3 = fit_convergence.DecodedW3;
    auto convergence_fv = fit_convergence.FitnessValue;
    cout << "Convergence - Decoded w1: " << convergence_decoded_w1 << endl;
    cout << "Convergence - Decoded w2: " << convergence_decoded_w2 << endl; 
    cout << "Convergence - Decoded w3: " << convergence_decoded_w3 << endl; 
    cout << "Convergence - Fitness Value: " << convergence_fv << endl;
    cout << "----------------------------------------------------------------" << endl;
    // decoded values and fitness value for best across generations
    auto fit_best_of_the_best = Fitness(best_of_the_best);
    auto best_of_the_best_decoded_w1 = fit_best_of_the_best.DecodedW1;
    auto best_of_the_best_decoded_w2 = fit_best_of_the_best.DecodedW2;
    auto best_of_the_best_decoded_w3 = fit_best_of_the_best.DecodedW3;
    auto best_of_the_best_fv = fit_best_of_the_best.FitnessValue;
    cout << "Best Of The Best - Decoded w1: " << best_of_the_best_decoded_w1 << endl;
    cout << "Best Of The Best - Decoded w2: " << best_of_the_best_decoded_w2 << endl; 
    cout << "Best Of The Best - Decoded w3: " << best_of_the_best_decoded_w3 << endl;
    cout << "Best Of The Best - Fitness Value: " << best_of_the_best_fv << endl;

    return 0;
}







