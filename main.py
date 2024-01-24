import yaml
class ModelGen(object):
    """Class for generate data, training and testing the specified model
    params----
    author : name of the author [Hu, Tariq, Sami] from the paper we are reimplementing the model
    dest_file_Path : where we want to save the generated data
    config : the configuration files with the required information. Edit to change the params

    modelFiles: path to save the trained models
    modelTestFiles: path to the test data"""
    def __init__(self, author, config="can_config.yaml"):
        self.author = author
        f = open(config,'r')
        self.configFile = yaml.safe_load(f)
        f.close()

        self.dest_file_path = self.configFile[author]['dataGen']['destfiles']
        self.datafilePath = self.configFile[author]['dataGen']['datafiles']

        self.modelfiles = self.configFile[author]['train']['modelFiles']
        self.testfiles = self.configFile[author]['test']['modelTestFiles']

        self.one_time_data_file = self.configFile[author]['dataGen']['one_time_test_data_files']
        self.one_time_dest_file = self.configFile[author]['dataGen']['one_time_test_dest_files']


    def generate_data(self):
        if self.author == 'Hu':
            import datagen_hu as dh
            print("Generating Data......")
            dh.gen_data(self.datafilePath, self.dest_file_path)
            print("Finished generation!")

    def generate_data_one_time(self):
        if self.author == 'Hu':
            import datagen_one_time_hu as dth
            print("Generating Data......")
            dth.gen_data(self.one_time_data_file, self.one_time_dest_file, self.configFile, self.author)
            print("Finished generation for one time test data!")

    def train_model(self):
        if self.author == 'Hu':
            import train_hu as th
            print("Training the model.....")
            th.train(self.modelfiles,self.testfiles,self.configFile,self.author,self.dest_file_path)

    def test_model(self):
        if self.author == 'Hu':
            import test_hu as ts
            print("Testing the model..")
            ts.test(self.modelfiles,self.testfiles)

    def test_model_one_time(self):
        if self.author == 'Hu':
            import test_one_time_hu as th
            print("Generating Data......")
            th.test(self.modelfiles, self.one_time_dest_file)
            

