for i in "symetric_gaussians" "disymmetric_gaussians" "cross"
do 
    python main_GMM_2D.py --N 100000 --metric_each_step True --name $i
done