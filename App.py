from AI.NeuralNetworkPackage.NeuralNetwork import NeuralNetwork
from AI.NeuralNetworkPackage.NeuralNetworkConstants import NeuralNetworkConstants as Const
from ImageConverter import ImageConverter
from os import listdir
from os.path import isfile, join
import time


class App:
    @staticmethod
    def create_tests():                           # creates tests input photos from row of numbers/sings
        my_path = Const.pre_test_folder           # folder with photos to cut out
        only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

        for path in only_files:
            image_converter = ImageConverter(save_images=1)
            input_data = image_converter.get_ai_input_data(my_path+"/"+path)

    @staticmethod
    def print_result(tmp):
        max = 0
        for j in range(14):
            if tmp[j] > tmp[max]:
                max = j
        if max == 0:
            second = 1
        else:
            second = 0
        for j in range(14):
            if tmp[j] > tmp[second] and j != max:
                second = j
        print(round(tmp[max] / sum(tmp) * 100, 2), end='% ')
        print(Const.names[max], end=" ")
        print(round(tmp[second] / sum(tmp) * 100, 2), end='% ')
        print(Const.names[second], end=" ")

    @staticmethod
    def learn_machine():
        start = time.time()
        training_path = Const.training_folder
        training_data_folders = [c for c in listdir(training_path)]  # get list of folders with data
        folder_nr = 0
        neural_network = NeuralNetwork(600, 400, 14)
        neural_network.load()
        for folder in training_data_folders:
            folder_path = training_path + "/" + folder
            training_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
            for file in training_files:
                file_path = folder_path + "/" + file
                training_data = ImageConverter.get_raw_data(path=file_path)
                neural_network.train(training_data, Const.training_result[folder_nr])
                print(file_path + " done.")

            print(folder_path + " done. Saving all changes to neural network.")
            neural_network.save()
            folder_nr += 1
        print("Time needed: +" + str(time.time() - start))

    @staticmethod
    def run(path):
        image_converter = ImageConverter(save_images=1)
        input_data = image_converter.get_ai_input_data(path)

        #  prepare objects for AI to work
        x = NeuralNetwork(600, 400, 14)
        x.load()
        start = time.time()
        for data in input_data:
            result = x.run(data)
            App.print_result(result)
            print("")
        print("Time needed: +" + str(time.time() - start))


#App.run("zdj.jpg")
App.learn_machine()
