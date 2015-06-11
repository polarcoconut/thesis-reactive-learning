import weka.core.jvm as jvm
from weka.datagenerators import DataGenerator
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random



jvm.start()
generator = DataGenerator(classname="weka.datagenerators.classifiers.classification.Agrawal", options=["-B", "-P", "0.05"])
DataGenerator.make_data(generator, ["-o", "testweka.arff"])
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("testweka.arff")

data.set_class_index(data.num_attributes() - 1) 
classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
evaluation = Evaluation(data)
evaluation.crossvalidate_model(classifier, data, 10, Random(42))
print(evaluation.to_summary())

print(evaluation.percent_correct())
