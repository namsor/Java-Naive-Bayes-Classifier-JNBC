/**
* A simple, scalable, explainable implementation of Naive Bayes Classifier.
* <ul>
* <li> NaiveBayesClassifierMapImpl works in-memory with a ConcurrentHashMap or off-the-heap using org.mapdb.HTreeMap
* <li> NaiveBayesClassifierMapLaplacedImpl adds Laplace smoothing to the implementation above
* <li> other popular Key-Value stores are supported : LevelDB and RocksDB
* <li> NaiveBayesExplainerImpl provides explainable trace of the algorithm, so it can be interpreted by human (formulae and expressions) or plain JavaScript
* </ul>
* Let's take the Sport/No Sport classif example. If the weather conditions are Sunny, Cool, Rainy (Humidity:High) and Windy then we are unlikely to play Sport. 
* <pre>{@code 
P(No)=0.795417348608838
P(Yes)=0.204582651391162
* }</pre>
* Firstly, we 'train' the classifier by calling the classifier.learn() method.
* <pre>{@code 
        String YES = "Yes";
        String NO = "No";
        String[] colName = {
            "outlook", "temp", "humidity", "wind", "play"
        };
        String[][] data = {
            {"Sunny", "Hot", "High", "Weak", "No"},
            {"Sunny", "Hot", "High", "Strong", "No"},
            {"Overcast", "Hot", "High", "Weak", "Yes"},
            {"Rain", "Mild", "High", "Weak", "Yes"},
            {"Rain", "Cool", "Normal", "Weak", "Yes"},
            {"Rain", "Cool", "Normal", "Strong", "No"},
            {"Overcast", "Cool", "Normal", "Strong", "Yes"},
            {"Sunny", "Mild", "High", "Weak", "No"},
            {"Sunny", "Cool", "Normal", "Weak", "Yes"},
            {"Rain", "Mild", "Normal", "Weak", "Yes"},
            {"Sunny", "Mild", "Normal", "Strong", "Yes"},
            {"Overcast", "Mild", "High", "Strong", "Yes"},
            {"Overcast", "Hot", "Normal", "Weak", "Yes"},
            {"Rain", "Mild", "High", "Strong", "No"},};

        String[] cats = {YES, NO};
        NaiveBayesClassifierMapImpl bayes = new NaiveBayesClassifierMapImpl("tennis", cats);
        for (int i = 0; i < data.length; i++) {
            Map<String, String> features = new HashMap();
            for (int j = 0; j < colName.length - 1; j++) {
                features.put(colName[j], data[i][j]);
            }
            bayes.learn(data[i][colName.length - 1], features);
        }
* }</pre>
* Secondly, 'predict' a value calling the classifier.classify() method with given weather conditions Sunny, Cool, Rainy and Windy.
* <pre>{@code 

        Map<String, String> features = new HashMap();
        features.put("outlook", "Sunny");
        features.put("temp", "Cool");
        features.put("humidity", "High");
        features.put("wind", "Strong");
        IClassification predict = bayes.classify(features, true);
        for (int i = 0; i < predict.getClassProbabilities().length; i++) {
            System.out.println("P(" + predict.getClassProbabilities()[i].getCategory() + ")=" + predict.getClassProbabilities()[i].getProbability());
        }
        if (predict.getExplanationData() != null) {
            NaiveBayesExplainerImpl explainer = new NaiveBayesExplainerImpl();
            IClassificationExplained explained = explainer.explain(predict);
            System.out.println(explained.toString());
        }
* }</pre>
* Finaly, 'explain' the value by calling the explainer.explain() method.
* The output details the likelyhood calculations as formulae and expressions that can be read by a human or by a Javascript interpreter.
* <pre>{@code 
* // JavaScript : 

// observation table variables 
var gL=14
var gL_cA_No=5
var gL_cA_No_fE_humidity=5
var gL_cA_No_fE_humidity_is_High=4
var gL_cA_No_fE_outlook=5
var gL_cA_No_fE_outlook_is_Sunny=3
var gL_cA_No_fE_temp=5
var gL_cA_No_fE_temp_is_Cool=1
var gL_cA_No_fE_wind=5
var gL_cA_No_fE_wind_is_Strong=3
var gL_cA_Yes=9
var gL_cA_Yes_fE_humidity=9
var gL_cA_Yes_fE_humidity_is_High=3
var gL_cA_Yes_fE_outlook=9
var gL_cA_Yes_fE_outlook_is_Sunny=2
var gL_cA_Yes_fE_temp=9
var gL_cA_Yes_fE_temp_is_Cool=3
var gL_cA_Yes_fE_wind=9
var gL_cA_Yes_fE_wind_is_Strong=3
var gL_fE_humidity=14
var gL_fE_outlook=14
var gL_fE_temp=14
var gL_fE_wind=14


// likelyhoods by category 

// likelyhoods for category No
var likelyhoodOfNo=gL_cA_No / gL * (gL_cA_No_fE_temp_is_Cool / gL_cA_No_fE_temp * gL_cA_No_fE_humidity_is_High / gL_cA_No_fE_humidity * gL_cA_No_fE_outlook_is_Sunny / gL_cA_No_fE_outlook * gL_cA_No_fE_wind_is_Strong / gL_cA_No_fE_wind * 1 )
var likelyhoodOfNoExpr=5 / 14 * (1 / 5 * 4 / 5 * 3 / 5 * 3 / 5 * 1 )
var likelyhoodOfNoValue=0.020571428571428574

// likelyhoods for category Yes
var likelyhoodOfYes=gL_cA_Yes / gL * (gL_cA_Yes_fE_temp_is_Cool / gL_cA_Yes_fE_temp * gL_cA_Yes_fE_humidity_is_High / gL_cA_Yes_fE_humidity * gL_cA_Yes_fE_outlook_is_Sunny / gL_cA_Yes_fE_outlook * gL_cA_Yes_fE_wind_is_Strong / gL_cA_Yes_fE_wind * 1 )
var likelyhoodOfYesExpr=9 / 14 * (3 / 9 * 3 / 9 * 2 / 9 * 3 / 9 * 1 )
var likelyhoodOfYesValue=0.005291005291005291


// probability estimates by category 

// probability estimate for category No
var probabilityOfNo=likelyhoodOfNo/(likelyhoodOfNo+likelyhoodOfYes+0)
var probabilityOfNoValue=0.795417348608838

// probability estimate for category Yes
var probabilityOfYes=likelyhoodOfYes/(likelyhoodOfNo+likelyhoodOfYes+0)
var probabilityOfYesValue=0.204582651391162


// return the highest probability estimate for evaluation 
probabilityOfNo
Result of evaluating mathematical expressions in String = 0.795417348608838

* }</pre>
* 
* @author elian
*/
package com.namsor.oss.classify.bayes;

