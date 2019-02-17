#Get the class name of max probility from csv file
def get_classfication_filename_from_testset():
    import pickle
    import csv

    results = {0:[], 1:[], 2:[], 3:[], 4:[],
               5:[], 6:[], 7:[], 8:[], 9:[]}

    #For submission_10_vgg16_20180724.csv, private score is 0.27117, public score is 0.28612
    with open('/home/ubuntu/distracted_driver_detection/subm/submission_10_vgg16_20180724.csv', 'r') as f:
        #reader = csv.reader(f)
        reader = csv.reader(f)
        i = 0
        for row in reader:
            #print(row)
            if i == 0:
                i += 1
                continue
            else:
                i += 1
                prob_value = list(map(float, row[1:]))
                #print(prob_value)
                key = 0
                max_value = max(prob_value)
                for value in prob_value:
                    if value == max_value:
                        results[key].append(row[0])
                    key += 1
                #break

    for k, v in results.items():
        print(k)
        print("len:{}".format(len(v)))
        
    f = open("/home/ubuntu/distracted_driver_detection/testset_result.pkl", "wb")
    pickle.dump(results, f)
    f.close()
    print("get_classfication_filename_from_testset_Done")
    
get_classfication_filename_from_testset()
