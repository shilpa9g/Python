import csv
import sys

#print "reading from file: ", sys.argv[1]
with open(sys.argv[1], 'r') as f:
    reader = csv.reader(f)
    my_input = list(reader)

#print "writing weights to file: ", sys.argv[2]
with open(sys.argv[2], 'w') as g:
    writer = csv.writer(g)

    # adding a bias 'b' of value 1 to each element
    for i, item in enumerate(my_input):
        item = list(map(int, item))
        item.insert(2, 1)
        my_input[i] = item
    #print(my_input)
    w = [0, 0, 0]
    w1 = [0,0,0]
    accr = 0

    def perceptron(data):
        for features in data:
            fx = w[0]*features[0] + w[1]*features[1] + w[2]*features[2]
            if features[3] * fx <= 0: # if y*f(x) <= 0
                for i in [0, 1, 2]:
                    w1[i] = w[i] + features[3] * features[i]
                writer.writerow(w1)
        return w1

    def accuracy(data):
        pos = 0
        for features in data:
            fx = w1[0]*features[0] + w1[1]*features[1] + w1[2]*features[2]
            if features[3] * fx > 0:
                pos += 1
        return pos
    
    runs = 0
    pCount = len(my_input)
    
    while runs < 100:
        runs += 1
        check = perceptron(my_input)
        pos = accuracy(my_input)
        print "Accuracy : " , pos , " / ", pCount
        
        if pos > accr:
            accr = pos
            w = check
            writer.writerow(w)
            print "Updated w: " , w
            
        if pos == pCount:
            break


