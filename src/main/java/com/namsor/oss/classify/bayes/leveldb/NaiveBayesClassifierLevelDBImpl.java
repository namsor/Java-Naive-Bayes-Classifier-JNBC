package com.namsor.oss.classify.bayes.leveldb;

import com.google.common.primitives.Longs;
import com.namsor.oss.classify.bayes.ClassificationImpl;
import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.IClassification;
import com.namsor.oss.classify.bayes.INaiveBayesClassifier;
import com.namsor.oss.classify.bayes.PersistentClassifierException;
import org.iq80.leveldb.ReadOptions;
import org.iq80.leveldb.WriteBatch;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Naive Bayes Classifier implementation with LevelDB as key/value store.
 * Learning is Synchronized but classification is not.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierLevelDBImpl extends AbstractNaiveBayesClassifierLevelDBImpl implements INaiveBayesClassifier {

    /**
     * Create a persistent Naive Bayes Classifier using LevelDB, with default cache size
     *
     * @param classifierName   The classifier name
     * @param categories       The immutable classification categories
     * @param rootPathWritable The writable directory for LevelDB storage
     * @throws PersistentClassifierException The persistence error and cause
     */
    public NaiveBayesClassifierLevelDBImpl(String classifierName, String[] categories, String rootPathWritable) throws PersistentClassifierException {
        super(classifierName, categories, rootPathWritable);
    }

    @Override
    public synchronized void learn(String category, Map<String, String> features, long weight) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.snapshot(getDb().getSnapshot());
        WriteBatch batch = getDb().createWriteBatch();
        try {
            String pathGlobal = pathGlobal();
            batch.put(bytes(pathGlobal), Longs.toByteArray((getDb().get(bytes(pathGlobal), ro) == null ? weight : Longs.fromByteArray(getDb().get(bytes(pathGlobal), ro)) + weight))); 
            String pathCategory = pathCategory(category);
            batch.put(bytes(pathCategory), Longs.toByteArray((getDb().get(bytes(pathCategory), ro) == null ? weight : Longs.fromByteArray(getDb().get(bytes(pathCategory), ro)) + weight)));
            for (Map.Entry<String, String> feature : features.entrySet()) {
                String pathFeatureKey = pathFeatureKey(feature.getKey());
                batch.put(bytes(pathFeatureKey), Longs.toByteArray((getDb().get(bytes(pathFeatureKey), ro) == null ? weight : Longs.fromByteArray(getDb().get(bytes(pathFeatureKey), ro)) + weight)));
                String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                batch.put(bytes(pathCategoryFeatureKey), Longs.toByteArray((getDb().get(bytes(pathCategoryFeatureKey), ro) == null ? weight : Longs.fromByteArray(getDb().get(bytes(pathCategoryFeatureKey), ro)) + weight)));
                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                batch.put(bytes(pathCategoryFeatureKeyValue), Longs.toByteArray((getDb().get(bytes(pathCategoryFeatureKeyValue), ro) == null ? weight : Longs.fromByteArray(getDb().get(bytes(pathCategoryFeatureKeyValue), ro)) + weight)));
            }
            getDb().write(batch);
        } finally {
            try {
                // Make sure you close the batch to avoid resource leaks.
                batch.close();
            } catch (IOException ex) {
                throw new ClassifyException(ex);
            }
            try {
                // Make sure you close the batch to avoid resource leaks.
                ro.snapshot().close();
            } catch (IOException ex) {
                throw new ClassifyException(ex);
            }
        }
    }

    @Override
    public IClassification classify(Map<String, String> features, final boolean explainData) throws ClassifyException {
        Map<String, Long> explanation = null;
        if (explainData) {
            explanation = new HashMap();
        }
        ReadOptions ro = new ReadOptions();
        ro.snapshot(getDb().getSnapshot());
        try {
            String pathGlobal = pathGlobal();
            long globalCount = (getDb().get(bytes(pathGlobal), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathGlobal), ro)));
            if (explainData) {
                explanation.put(pathGlobal, globalCount);
            }
            double[] likelyhood = new double[getCategories().length];
            double likelyhoodTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                String pathCategory = pathCategory(category);
                long categoryCount = (getDb().get(bytes(pathCategory), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathCategory), ro)));
                if (explainData) {
                    explanation.put(pathCategory, categoryCount);
                }
                double product = 1.0d;
                for (Map.Entry<String, String> feature : features.entrySet()) {
                    String pathFeatureKey = pathFeatureKey(feature.getKey());
                    //double featureCount = (getDb().get(ro, bytes(pathFeatureKey)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKey))));
                    long featureCount = (getDb().get(bytes(pathFeatureKey), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathFeatureKey), ro)));
                    if (explainData) {
                        explanation.put(pathFeatureKey, featureCount);
                    }
                    if (featureCount > 0) {
                        String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                        long categoryFeatureCount = (getDb().get(bytes(pathCategoryFeatureKey), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathCategoryFeatureKey), ro)));
                        if (explainData) {
                            explanation.put(pathCategoryFeatureKey, categoryFeatureCount);
                        }
                        String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                        long categoryFeatureValueCount = (getDb().get(bytes(pathCategoryFeatureKeyValue), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathCategoryFeatureKeyValue), ro)));
                        if (explainData) {
                            explanation.put(pathCategoryFeatureKeyValue, categoryFeatureValueCount);
                        }
                        double basicProbability = (categoryFeatureCount == 0 ? 0 : 1d * categoryFeatureValueCount / categoryFeatureCount);
                        product *= basicProbability;
                    }
                }
                likelyhood[i] = 1d * categoryCount / globalCount * product;
                likelyhoodTot += likelyhood[i];
            }
            return new ClassificationImpl(features, likelihoodsToProbas(likelyhood, likelyhoodTot), explanation, likelyhoodTot);
        } finally {
            try {
                // Make sure you close the snapshot to avoid resource leaks.
                ro.snapshot().close();
            } catch (IOException ex) {
                throw new ClassifyException(ex);
            }
        }
    }

}
