package com.namsor.oss.classify.bayes;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;

/**
 * A simple, scalable Naive Bayes Classifier, based on a key-value store (in
 * memory, or disk-based)
 *
 * @author elian carsenat, NamSor SAS
 */
public abstract class AbstractNaiveBayesClassifierImpl extends AbstractNaiveBayesImpl implements INaiveBayesClassifier {

    private final String classifierName;
    private final String[] categories;

    public AbstractNaiveBayesClassifierImpl(String classifierName, String[] categories) {
        this.classifierName = classifierName;
        this.categories = categories;
    }

    /**
     * @return the classifierName
     */
    public String getClassifierName() {
        return classifierName;
    }

    /**
     * @return the categories
     */
    @Override
    public String[] getCategories() {
        return categories;
    }

    protected final Comparator<IClassProbability> orderByProba = new Comparator() {
        @Override
        public int compare(Object o1, Object o2) {
            IClassProbability c1 = (IClassProbability) o1;
            IClassProbability c2 = (IClassProbability) o2;
            return ((Double) c2.getProbability()).compareTo((Double) c1.getProbability());
        }
    };

    @Override
    public synchronized void learn(String category, Map<String, String> features) throws ClassifyException {
        learn(category, features, 1);
    }

    protected IClassProbability[] likelihoodsToProbas(double[] likelyhood, double likelyhoodTot) {
        IClassProbability[] result = new ClassProbabilityImpl[getCategories().length]; // TODO : if likelyhoodTot=0 then
        for (int i = 0; i < getCategories().length; i++) {
            double proba;
            if (likelyhoodTot <= 0) {
                // this is an underflow/overflow error
                proba = 0d;
            } else {
                proba = likelyhood[i] / likelyhoodTot;
                if (proba > 1d) {
                    // could equal 1.000000000002 due to double precision issue;
                    proba = 1d;
                } else if (proba < 0) {
                    proba = 0d;
                }
            }
            ClassProbabilityImpl classif = new ClassProbabilityImpl(getCategories()[i], proba);
            result[i] = classif;
        }
        Arrays.sort(result, orderByProba);
        return result;
    }
}
