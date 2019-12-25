package com.namsor.oss.samples;

import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.IClassification;
import com.namsor.oss.classify.bayes.INaiveBayesClassifier;
import com.namsor.oss.classify.bayes.NaiveBayesClassifierMapLaplacedImpl;
import java.io.IOException;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Simple test. Can you find features that will help predict if a number is prime (based on the previous prime) ? Probably not.
 * https://www.independent.co.uk/news/science/maths-experts-stunned-as-they-crack-a-pattern-for-prime-numbers-a6933156.html
 * @author elian
 */
public class MainSample4 {

    private final int size;
    private final double testRatio;
    private final boolean[] primes;
    private final INaiveBayesClassifier classifier;

    private static final String PRIME = "Prime";
    private static final String NOT_PRIME = "NotPrime";
    private static final String[] CATS = {PRIME, NOT_PRIME};

    public MainSample4(int size, double testRatio) {
        this.size = size;
        this.testRatio = testRatio;
        this.primes = primes(size);
        this.classifier = new NaiveBayesClassifierMapLaplacedImpl("sentiment", CATS);
    }

    private static boolean[] primes(int size) {
        boolean[] primes = new boolean[size];
        Arrays.fill(primes, true);
        for (int p = 2; p * p <= size; p++) {
            if (primes[p]) {
                for (int i = p * 2; i < size; i += p) {
                    primes[i] = false;
                }
            }
        }
        return primes;
    }

    public List<Integer> allPrimes() {
        List<Integer> primeNumbers = new LinkedList<>();
        for (int i = 2; i < size; i++) {
            if (primes[i]) {
                primeNumbers.add(i);
            }
        }
        return primeNumbers;
    }

    private static void appendNGrams(Map<String, String> features, String featureType, int maxLen, String namePart, boolean inside) {
        if (namePart.length() > maxLen) {
            for (int i = 1; i <= maxLen; i++) {
                features.put("#" + featureType + "_^" + i, namePart.substring(0, i) + "*");
            }
            for (int i = 1; i <= maxLen; i++) {
                features.put("#" + featureType + "_*" + i, namePart.substring(namePart.length() - i, namePart.length()) + "$");
            }
            if (inside) {
                for (int i = 2; i <= maxLen; i++) {
                    for (int j = 1; j < namePart.length() - 1; j++) {
                        if (j + i < namePart.length()) {
                            features.put("#" + featureType + "_*" + i + "x" + j + "*", namePart.substring(j, j + i) + "*");
                        }
                    }
                }
            }
        } else {
            for (int i = 1; i <= namePart.length(); i++) {
                features.put("#" + featureType + "_^" + i, namePart.substring(0, i) + "*");
            }
            for (int i = 1; i <= namePart.length(); i++) {
                features.put("#" + featureType + "_*" + i, namePart.substring(namePart.length() - i, namePart.length()) + "$");
            }
        }
        features.put("#" + featureType, "" + namePart);
    }

    public Map<String, String> features(long previousPrime, long number) {
        Map<String, String> features = new HashMap();
        appendNGrams(features, "P-1", 2, "" + (previousPrime), false);
        appendNGrams(features, "N", 2, "" + number, false);
        return features;
    }

    public static String toString(Map<String, String> features) {
        StringWriter sw = new StringWriter();
        for (Map.Entry<String, String> entry : features.entrySet()) {
            sw.append(entry.getKey() + "=" + entry.getValue() + "\n");
        }
        return sw.toString();
    }

    private final boolean SKIP_TRIVIAL = true;

    private void train() {
        try {
            int trainSize = (int) (size * (1d - testRatio));
            int previousPrime = 1;
            int train = 0;
            for (int i = 2; i < trainSize; i++) {
                int lastDigit = i % 10;
                if (SKIP_TRIVIAL && !(lastDigit == 1 || lastDigit == 3 || lastDigit == 7 || lastDigit == 9)) {
                    continue;
                }
                Map<String, String> features = features(previousPrime, i);
                String primeOrNot = (primes[i] ? PRIME : NOT_PRIME);
                if (primes[i]) {
                    previousPrime = i;
                }
                classifier.learn(primeOrNot, features);
                train++;
                if ((train % 1000 == 0 && train < 10000)
                        || (train % 10000 == 0 && train < 100000)
                        || (train % 100000 == 0 && train < 1000000)
                        || train % 1000000 == 0) {
                    Logger.getLogger(getClass().getName()).info("train = "+train+" i = " + i);// + " features = \n" + toString(features));
                }
            }
        } catch (ClassifyException ex) {
            Logger.getLogger(MainSample4.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void test() {
        try {
            int trainSize = (int) (size * (1d - testRatio));
            int[][] confusion = {
                {0, 0},
                {0, 0},};
            int ok = 0;
            int ko = 0;
            int previousPrime = 3;
            int test = 0;
            for (int i = trainSize; i < size; i++) {
                int lastDigit = i % 10;
                if (SKIP_TRIVIAL && !(lastDigit == 1 || lastDigit == 3 || lastDigit == 7 || lastDigit == 9)) {
                    continue;
                }
                Map<String, String> features = features(previousPrime, i);
                String primeOrNot = (primes[i] ? PRIME : NOT_PRIME);
                int x = (primes[i] ? 0 : 1);
                IClassification[] classif = classifier.classify(features);
                if (primes[i]) {
                    previousPrime = i;
                }
                int y = (classif[0].getCategory().equals(PRIME) ? 0 : 1);
                //System.out.println(i+"|"+primeOrNot+"|"+classif[0].getCategory()+"|"+classif[0].getProbability());
                confusion[x][y]++;
                if (primeOrNot.equals(classif[0].getCategory())) {
                    ok++;
                } else {
                    ko++;
                }
                test++;
                if ((test % 1000 == 0 && test < 10000)
                        || (test % 10000 == 0 && test < 100000)
                        || (test % 100000 == 0 && test < 1000000)
                        || test % 1000000 == 0) {
                    Logger.getLogger(getClass().getName()).info("test = "+test+" i = " + i + " Ok = " + ok + " Ko = " + ko + " Confusion = \n" + confusion[0][0] + "\t" + confusion[0][1] + "\n" + confusion[1][0] + "\t" + confusion[1][1] + "\n");
                }

            }
            Logger.getLogger(getClass().getName()).info("Finally, test = "+test+" Ok = " + ok + " Ko = " + ko + " r = "+(ok/(1d*ok+ko))+ " Confusion = \n" + confusion[0][0] + "\t" + confusion[0][1] + "\n" + confusion[1][0] + "\t" + confusion[1][1] + "\n");
        } catch (ClassifyException ex) {
            Logger.getLogger(MainSample4.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static final void main(String[] args) {
        MainSample4 main = new MainSample4(10000000, 0.5);
        System.out.println(main.toString(main.features(3, 5)));
        main.train();
        main.test();

    }

}
