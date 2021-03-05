

"meta-learning-optimizers" is a python library to learn black-box optimization algorithms from scratch in a data-driven way. 

Instead of thinking on each piece of a particular algorithm someone might need one day to solve a given problem, we learn the optimizer using a distribution of tasks.



The LTO-POMDP is implemented in the folder "meta-learning-optimizers/environments/learning_to_optimize_POMDP/"

The policies used in the paper are implemented in the folder meta-learning-optimizers/policies/"

The meta-optimizers used in the paper are implemented in the folder meta-learning-optimizers/optimizers/"

Then, we can run all the experiments based on "runners" in the folder meta-learning-optimizers/runners/"


To run LTO-POMDP a particular experiment you need:
    * create an cfg file in .YAML format. An example is provided in cfg folder.
    * use main.py to run this cfg file

There are two possible plots to visualize: ecdfs and meta-losses

However, we need to train the optimizer and run the baselines. To this end, it is required to run the following 

To improve the optimizer in the training environment:
$ mpiexec -n NB_CPUS python3 main.py --config_path cfg/examples/1learning_train.yaml

To run the optimizer on the validation environment:
$ python3 main.py --config_path cfg/examples/1learning_validation.yaml

To run the optimizer on the test environment:
$ python3 main.py --config_path cfg/examples/2preprocessing_test_policy.yaml

To run CMA-ES in the training, validation and test environment:
$ python3 main.py --config_path cfg/examples/2preprocessing_train_cmaes.yaml
$ python3 main.py --config_path cfg/examples/2preprocessing_val_cmaes.yaml
$ python3 main.py --config_path cfg/examples/2preprocessing_test_cmaes.yaml

To run Random Search in the training, validation and test environment:
$ python3 main.py --config_path cfg/examples/2preprocessing_train_rs.yaml
$ python3 main.py --config_path cfg/examples/2preprocessing_val_rs.yaml
$ python3 main.py --config_path cfg/examples/2preprocessing_test_rs.yaml

To plot the results as ECDFs:
$ python3 main.py --config_path cfg/examples/3postprocessing_ecdfs.yaml

To plot the results as metalosses:
$ python3 main.py --config_path cfg/examples/3postprocessing_metalosses.yaml

This will save an "ecdf.pdf" and "metaloss.pdf" :)