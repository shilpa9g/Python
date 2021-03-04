import csv
import sys
#import statistics
import numpy as np

#with open(sys.argv[1], 'r') as f:
with open('input2.csv', 'r') as f:
    reader = csv.reader(f)
    my_input = list(reader)

#with open(sys.argv[2], 'w') as g:
with open('output2.csv', 'w') as g:
    writer = csv.writer(g)

     # adding a bias 'b' of value 1 to each element
    for i, item in enumerate(my_input):
        item.insert(0, 1)
        item = list(map(float, item))
        #mean_age = statistics.mean(my_input[1])
        my_input[i] = item
    print(my_input)

    arr = np.array(my_input)
    #ages = reader.iloc[:,1]
    #weights = reader.iloc[:,2]
    #plt.scatter(ages, weights)
    #plt.show
    mean_age = arr[:,1].mean()
    stdev_age = arr[:,1].std()
    mean_weight = arr[:,2].mean()
    stdev_weight = arr[:,2].std()
    scaled_arr = arr

    for i in range(len(my_input)):
        scaled_arr[i][1] = (arr[i][1] - mean_age)/stdev_age
        scaled_arr[i][2] = (arr[i][2] - mean_weight)/stdev_weight
    print(scaled_arr)

    w = [0, 0, 0]
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.0001]
    epochs = 100
    n = len(my_input)
    

    def gradient_descent(data, alpha):
        add_sq_diff = 0
        b0_sum = 0
        b1_sum = 0
        b2_sum = 0
        for features in data:
            fx = w[0]*features[0] + w[1]*features[1] + w[2]*features[2]
            sq_diff = (fx - features[3]) ** 2
            add_sq_diff += sq_diff
            b0 = (fx - features[3]) * features[0]
            b0_sum += b0
            b1 = (fx - features[3]) * features[1]
            b1_sum += b1
            b2 = (fx - features[3]) * features[2]
            b2_sum += b2
        loss = add_sq_diff/(2*n)
        w[0] = w[0] - (alpha * b0_sum)/n
        w[1] = w[1] - (alpha * b1_sum)/n
        w[2] = w[2] - (alpha * b2_sum)/n
        return loss

    old_loss = 10
    converge = {}
    for i, alpha in enumerate(alphas):
        w = [0, 0, 0]
        for epoch in range(epochs):
            loss = gradient_descent(scaled_arr, alpha)
            if loss < old_loss:
                if loss == 0:
                    converge[i] = epoch + 1
        output = [alpha, epochs, w[0], w[1], w[2]]
        writer.writerow(output)
    #print(converge)
