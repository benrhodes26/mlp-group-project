# This script is a re-implementation of some of the pre-processing used
# here: https://github.com/siyuanzhao/2016-EDM/. The raw csv data used here
# can be downloaded from that repo.
# The output of this script is fed into the data_providers class, which
# does some further processing to construct mini-batches for training an RNN

import csv
import numpy as np
import os
import scipy.sparse as sp
import sys

data_dir = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

input_data_path = os.path.join(data_dir, input_filename)
output_data_path = os.path.join(data_dir, output_filename)

num_lines_per_student = 3
num_students = 0
max_num_ans = 0  # largest number of questions answered by any student
max_prob_set_id = 0  # largest id of any problem set
student_to_prob_sets = {}
student_to_marks = {}
prob_set_counts = {}

# every 3 lines contains the data for a new student.
# line 1: number of problems attempted
# line 2: sequence of problem_set_ids of the problems attempted
# line 3: corresponding sequence of marks (1=correct, 0=incorrect)

with open(input_data_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    skip = False  # we will skip any student who has answered fewer than 3 problems
    for i, row in enumerate(reader):
        row = list(map(int, row))

        if i % num_lines_per_student == 0:
            # row contains number of problems student has answered
            if row[0] < 3:
                skip = True
            else:
                skip = False
                num_students += 1
                max_num_ans = max(max_num_ans, row[0])

        if i % num_lines_per_student == 1 and not skip:
            # row contains list of problem set ids, one for each problem student answered
            for prob_set in row:
                prob_set_counts[str(prob_set+1)] = prob_set_counts.get(str(prob_set+1   ), 0) + 1
            # store the problem ids for this student, adding 1 so the ids start from 1, not 0
            student_to_prob_sets[str(num_students)] = [i+1 for i in row]

        if i % num_lines_per_student == 2 and not skip:
            # row contains list of 0/1s meaning wrong/right answer was given to
            # corresponding problem in previous row
            student_to_marks[str(num_students)] = row

max_prob_set_id = max(map(int, prob_set_counts.keys()))
encoding_dim = (2 * max_prob_set_id) + 1

# We would now like to build a matrix, where each row corresponds to a student
# and the row contains a concatenated sequence of one-hot-encoded vectors indicating the
# sequence of problems a student attempted and whether or not they answered correctly.
# This matrix will mostly contain zeros and a only a few ones.
# Hence, we only store the coordinates of the ones and then create a sparse matrix
# using scipy.sparse library.

row_coordinates = []
column_coordinates = []
# target_ids = []
target_ids_row_coords = []
target_ids_col_coords = []
targets = []
for i in range(num_students):
    problems = student_to_prob_sets[str(i+1)]
    marks = student_to_marks[str(i+1)]
    num_problems = len(problems) - 1  # we can't make predictions for first problem, so ignore

    # for each problem (except the last) a student has answered, generate an index
    # specifying the index of the 1 in a 1-hot encoded vector.
    # this vector both encodes the problem_set_id AND whether or not the student
    # answered correctly. Hence the vec tor has length: 2*max_prob_set_id
    make_1_hot_index = lambda x: (1 - x[1])*x[0] + x[1]*(max_prob_set_id + x[0])
    encoding_indices = np.array(list(map(make_1_hot_index, zip(problems[:-1], marks[:-1]))))

    # we want to concatenate the above 1 hot vectors in order to obtain a very long feature vector of
    # 0s and 1s. We then want to right pad this vector with zeros, such that all students have
    # the same length vector.
    # Instead of storing this whole feature vector, we store the indices of the 1s.
    indices_of_ones = encoding_dim*np.arange(num_problems) + encoding_indices

    # We now have the 'coordinates' of the ones in student i's feature vector.
    # Let's now add to a global set of coordinates for all students
    row_coordinates.extend(list(i*np.ones(num_problems)))
    column_coordinates.extend(list(indices_of_ones))

    # calculate target_ids that we need after learning to extract a predictions
    # vector for each student that corresponds to the targets vector
    target_ids_row_coords.extend(list(i*np.ones(num_problems)))
    target_ids_col_coords.extend(list(max_prob_set_id*np.arange(num_problems) + problems[1:] - 1))

    # add targets. Exclude first mark (since we have nothing to predict it with)
    targets.append(marks[1:])

sparse_inputs = sp.csr_matrix((np.ones(len(row_coordinates)), (row_coordinates, column_coordinates)),
                              shape=(num_students, max_num_ans*encoding_dim))
sparse_target_ids = sp.csr_matrix((np.ones(len(target_ids_row_coords)), (target_ids_row_coords, target_ids_col_coords)),
                                  shape=(num_students, max_num_ans*max_prob_set_id))

inputs_data_path = output_data_path + '-inputs'
target_ids_data_path = output_data_path + '-targetids'
targets_data_path = output_data_path + '-targets'

sp.save_npz(inputs_data_path, sparse_inputs)
sp.save_npz(target_ids_data_path, sparse_target_ids)
np.savez(targets_data_path, targets=np.array(targets),
         max_num_ans=max_num_ans, max_prob_set_id=max_prob_set_id)

# DATA CHECKING
# write problem set ids and counts to a file for visual inspection
unique_prob_sets = '\n'.join(list(map(str, sorted(prob_set_counts.items(), key=lambda x: int(x[0])))))
with open('{}/unique-prob-set-counts-{}.txt'.format(data_dir, output_filename), "w") as f:
    f.write('problem set id, count \n')
    f.write(unique_prob_sets)

# print out summary data
print(
    'There are {} students. \n'
    'The max number of questions answered by any student is {}. \n'
    'The max id of a problem set is {}.'.format(num_students, max_num_ans, max_prob_set_id)
      )