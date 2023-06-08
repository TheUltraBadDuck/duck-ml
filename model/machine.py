import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Machine():
    
    # * - - - Constructor - - -
    def __init__(self, r_file: str = "") -> None:

        try:
            if len(r_file) == 0:
                pass
            elif r_file.endswith(".csv"):
                self.df = pd.read_csv(r_file)
            elif r_file.endswith(".xls") or r_file.endswith(".xlsx"):
                self.df = pd.read_excel(r_file)
            elif r_file.endswith(".json"):
                self.df = pd.read_json(r_file)
            else:
                raise("[INIT ERROR]: Cannot read such file ", r_file)
        except:
            raise("[INIT ERROR]: Cannot read such file ", r_file)

        self.r_file: str = r_file
        self.w_file: str = ""

        self.df_train: pd = self.df
        self.df_test:  pd = None

        self.shape: tuple = self.df.shape
        self.size: int = self.df.size
    
        self.schema: list = []
        for col in self.df.columns:
            self.schema.append(col)



    # * - - - Convert non-number values to int - - -
    def numerize(self, col_names = None):

        if not self.df_test is None:
            raise("[NUMERIZATION ERROR]: Need to numerize data before splitting")

        self.numerization: dict = {}

        if col_names is None:
            col_names = self.df.column

        for col in col_names:

            if (self.df[col] is int) or (self.df[col] is float):
                continue
            new_val: int = 0
            self.numerization[col] = {}

            for c in range(len(self.df[col])):

                old_val = self.df.loc[c, col]
                if not old_val in self.numerization[col]:
                    self.numerization[col][old_val] = new_val
                    new_val += 1
                self.df.loc[c, col] = self.numerization[col][old_val]
    


    # * - - - Modify data - - -
    def modify(self, col: str, func):

        if not self.df_test is None:
            raise("[MODIFICATION ERROR]: Need to modify data before splitting")

        for c in range(len(self.df[col])):
            self.df.loc[c, col] = func(self.df.loc[c, col])



    def info(self, write: str = ""):

        print_type: str = ""
        parameter = None

        if write == "train":
            print_type = "TRAIN"
            parameter = self.df_train.head()

        elif write == "test":
            print_type = "TEST"
            parameter = self.df_test.head()

        elif (write == "train_X") or (write == "X_train"):
            print_type = "TRAIN X"
            parameter = self.X_train[:5, :]

        elif (write == "train_y") or (write == "y_train"):
            print_type = "TRAIN Y"
            parameter = self.y_train[:5, :]
        
        elif (write == "test_X") or (write == "X_test"):
            print_type = "test X"
            parameter = self.X_test[:5, :]

        elif (write == "test_y") or (write == "y_test"):
            print_type = "test Y"
            parameter = self.y_test[:5, :]
        
        print(print_type, ":")
        print(" >  shape: ", parameter.shape)
        print(parameter)
        




class Regression(Machine):

    def __init__(self, r_file: str = "") -> None:
        Machine.__init__(self, r_file)



    def split(self, train_scale: float, test_scale: float, seed: int = 0):

        if (train_scale <= 0) or (test_scale <= 0):
            raise("[SPLIT ERROR]: Contains negative or zero ratio")

        ratio: float = train_scale / (train_scale + test_scale)
        cut_pos: int = int(ratio * self.shape[0])

        if seed != -1:
            np.random.seed(seed)
            id_list: np = np.random.permutation((self.shape[0]))

            self.df_train = self.df.loc[id_list[:cut_pos], :]
            self.df_test = self.df.loc[id_list[cut_pos:], :]
        else:
            self.df_train = self.df.loc[:cut_pos, :]
            self.df_test = self.df.loc[cut_pos:, :]



    def giveTheory(self, features, labels):

        self.schema_X: list = features
        self.schema_y: list = labels

        self.X_train: np = self.df_train.loc[:, features].to_numpy()
        self.y_train: np = self.df_train.loc[:, labels].to_numpy()
        self.X_test: np = self.df_test.loc[:, features].to_numpy()
        self.y_test: np = self.df_test.loc[:, labels].to_numpy()

        if len(np.shape(self.y_train)) == 1:
            self.y_train = np.reshape(self.y_train, (-1, 1))
            self.y_test = np.reshape(self.y_test, (-1, 1))



    def charge(self, file_name: str = "", key: str = ""):
        pass



    def graph(self, x_title: str, y_title: str, write_arr: list):

        for write in write_arr:

            id_x: int = self.schema_X.index(x_title)
            id_y: int = self.schema_y.index(y_title)
            if (id_x == -1) or (id_y == -1):
                raise("!!!!!!!!!!!!!!!!!!!")
            
            if write == "both":
                plt.scatter(self.df.loc[:, x_title], self.df.loc[:, y_title], color = "black")

            if write == "train":
                plt.scatter(self.X_train[:, id_x], self.y_train[:, id_y], color = "blue")

            elif write == "test":
                plt.scatter(self.X_test[:, id_x], self.y_test[:, id_y], color = "orange")
            
            elif write == "pred":
                plt.scatter(self.X_test[:, id_x], self.y_pred[:, id_y], color = "red")

        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()



    def info(self, write: str):

        print_type: str = ""
        parameter = None

        if write == "train":
            print_type = "TRAIN"
            parameter = self.df_train.head()

        elif write == "test":
            print_type = "TEST"
            parameter = self.df_test.head()

        elif (write == "train_X") or (write == "X_train"):
            print_type = "TRAIN X"
            parameter = self.X_train[:5, :]

        elif (write == "train_y") or (write == "y_train"):
            print_type = "TRAIN Y"
            parameter = self.y_train[:5, :]
        
        elif (write == "test_X") or (write == "X_test"):
            print_type = "test X"
            parameter = self.X_test[:5, :]

        elif (write == "test_y") or (write == "y_test"):
            print_type = "test Y"
            parameter = self.y_test[:5, :]
        
        print(print_type, ":")
        print(" >  shape: ", parameter.shape)
        print(parameter)


