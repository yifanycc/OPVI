
## test for increasing bath-size Nov 11
#python main_batch.py --T 50 --batchType linear --stepType 1t >> log_nov11.txt
#python main_batch.py --T 50 --batchType sub --stepType 1t >> log_nov11.txt
#python main_batch.py --T 50 --batchType linear --stepType 0.1t >> log_nov11.txt
#python main_batch.py --T 50 --batchType sub --stepType 0.1t >> log_nov11.txt
#
#
#python main_stochastic_full.py --T 500 --stepType 1t >> log_nov11.txt
#python main_stochastic_full.py --T 500 --stepType 0.1t >> log_nov11.txt
#python main_stochastic_full.py --T 500 --stepType 0.1 >> log_nov11.txt
#python main_stochastic_full.py --T 500 --stepType 0.01 >> log_nov11.txt
#
#

## change c1, c2
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.1 --c3 6 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.1 --c3 6  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.1 --c3 6  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.01 --c3 6 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.01 --c3 6  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.01 --c3 6  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.001 --c3 6 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.001 --c3 6  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.001 --c3 6  >> log_dec6.txt
#
## change c3
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.1 --c3 4 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.1 --c3 4  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.1 --c3 4  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.01 --c3 4 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.01 --c3 4  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.01 --c3 4  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.001 --c3 4 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.001 --c3 4  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.001 --c3 4  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.1 --c3 2 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.1 --c3 2  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.1 --c3 2  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.01 --c3 2 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.01 --c3 2  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.01 --c3 2  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.1 --c2 0.001 --c3 2 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.01 --c2 0.001 --c3 2  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 1t --c1 0.001 --c2 0.001 --c3 2  >> log_dec6.txt
#
#
#
## change stepsize
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.1 --c3 6 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.1 --c3 6  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.1 --c3 6  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.01 --c3 6 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.01 --c3 6  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.01 --c3 6  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.001 --c3 6 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.001 --c3 6  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.001 --c3 6  >> log_dec6.txt
#
## change c3
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.1 --c3 4 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.1 --c3 4  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.1 --c3 4  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.01 --c3 4 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.01 --c3 4  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.01 --c3 4  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.001 --c3 4 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.001 --c3 4  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.001 --c3 4  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.1 --c3 2 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.1 --c3 2  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.1 --c3 2  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.01 --c3 2 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.01 --c3 2  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.01 --c3 2  >> log_dec6.txt
#
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.1 --c2 0.001 --c3 2 >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.01 --c2 0.001 --c3 2  >> log_dec6.txt
#python main_stochastic_full.py --T 1000 --stepType 0.1t --c1 0.001 --c2 0.001 --c3 2  >> log_dec6.txt
#


# test for batch take away (sample size 10000 / 5000)
python main_batch.py --T 1000 --stepType 1t --c1 0.1 --c2 0.01 --c3 6 >> log_dec9.txt
python main_batch.py --T 1000 --stepType 1t --c1 0.2 --c2 0.02 --c3 6  >> log_dec9.txt
python main_batch.py --T 1000 --stepType 1t --c1 0.1 --c2 0.02 --c3 6  >> log_dec9.txt


python main_batch.py --T 250 --stepType 1t --c1 0.1 --c2 0.01 --c3 6 >> log_dec9.txt
python main_batch.py --T 250 --stepType 1t --c1 0.2 --c2 0.02 --c3 6  >> log_dec9.txt
python main_batch.py --T 250 --stepType 1t --c1 0.1 --c2 0.02 --c3 6  >> log_dec9.txt

# test for batch constant
python main_batch.py --T 1000 --stepType 1t --c1 0.1 --c2 0.0 --c3 6 >> log_dec9.txt
python main_batch.py --T 1000 --stepType 1t --c1 0.2 --c2 0.0 --c3 6  >> log_dec9.txt


python main_batch.py --T 250 --stepType 1t --c1 0.1 --c2 0.0 --c3 6 >> log_dec9.txt
python main_batch.py --T 250 --stepType 1t --c1 0.2 --c2 0.0 --c3 6  >> log_dec9.txt

# test for stochastic take away
python main.py --T 20000 --stepType 1t --c1 0.1 --c2 0.01 --c3 6 >> log_dec9.txt
python main.py --T 20000 --stepType 1t --c1 0.01 --c2 0.001 --c3 6  >> log_dec9.txt
python main.py --T 20000 --stepType 1t --c1 0.05 --c2 0.005 --c3 6  >> log_dec9.txt


python main.py --T 5000 --stepType 1t --c1 0.1 --c2 0.01 --c3 6 >> log_dec9.txt
python main.py --T 5000 --stepType 1t --c1 0.01 --c2 0.001 --c3 6  >> log_dec9.txt
python main.py --T 5000 --stepType 1t --c1 0.05 --c2 0.005 --c3 6  >> log_dec9.txt

# test for stochastic constant

python main_batch.py --T 20000 --stepType 1t --c1 0.1 --c2 0.0 --c3 6 >> log_dec9.txt
python main_batch.py --T 20000 --stepType 1t --c1 0.01 --c2 0.0 --c3 6  >> log_dec9.txt


python main_batch.py --T 5000 --stepType 1t --c1 0.1 --c2 0.0 --c3 6 >> log_dec9.txt
python main_batch.py --T 5000 --stepType 1t --c1 0.01 --c2 0.0 --c3 6  >> log_dec9.txt

exit 0