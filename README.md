# Automatic_fitting_genetic
Automatic fitting of variograms using genetic algorithms

This algorithm use standart gslib output for modelling experimental variograms 
using genetic algorithms. This aproach consist in create several individuals with their 
properties such range, sill, contribution, and nugget effect. The population select 
the best individual with an objective function that minimize the square difference 
among the model and experimentals. The best model is returned past some 
generations. 
