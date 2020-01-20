package com.namsor.oss.classify.bayes.rocksdb;

import com.google.common.primitives.Longs;
import com.namsor.oss.classify.bayes.ClassificationImpl;
import com.namsor.oss.classify.bayes.ClassifyException;
import com.namsor.oss.classify.bayes.IClassification;
import com.namsor.oss.classify.bayes.INaiveBayesClassifier;
import com.namsor.oss.classify.bayes.PersistentClassifierException;
import org.rocksdb.ReadOptions;
import org.rocksdb.RocksDBException;
import org.rocksdb.WriteBatch;
import org.rocksdb.WriteOptions;

import java.util.HashMap;
import java.util.Map;

/**
 * Naive Bayes Classifier implementation with RocksDB as key/value store.
 * Learning is Synchronized but classification is not.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierRocksDBImpl extends AbstractNaiveBayesClassifierRocksDBImpl implements INaiveBayesClassifier {

    /**
     * Create a Naive Bayes Classifier implementation with RocksDB as key/value store.
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param rootPathWritable A writable path for RocksDB
     * @throws PersistentClassifierException The persistence error and cause
     */
    public NaiveBayesClassifierRocksDBImpl(String classifierName, String[] categories, String rootPathWritable) throws PersistentClassifierException {
        super(classifierName, categories, rootPathWritable);
    }

    @Override
    public synchronized void learn(String category, Map<String, String> features, long weight) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(getDb().getSnapshot());
        WriteOptions wo = new WriteOptions();
        WriteBatch batch = new WriteBatch();
        try {
            String pathGlobal = pathGlobal();
            batch.put(bytes(pathGlobal), Longs.toByteArray((getDb().get(ro, bytes(pathGlobal)) == null ? weight : Longs.fromByteArray(getDb().get(ro, bytes(pathGlobal))) + weight)));
            String pathCategory = pathCategory(category);
            batch.put(bytes(pathCategory), Longs.toByteArray((getDb().get(ro, bytes(pathCategory)) == null ? weight : Longs.fromByteArray(getDb().get(ro, bytes(pathCategory))) + weight)));
            for (Map.Entry<String, String> feature : features.entrySet()) {
                String pathFeatureKey = pathFeatureKey(feature.getKey());
                batch.put(bytes(pathFeatureKey), Longs.toByteArray((getDb().get(ro, bytes(pathFeatureKey)) == null ? weight : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKey))) + weight)));
                String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                batch.put(bytes(pathCategoryFeatureKey), Longs.toByteArray((getDb().get(ro, bytes(pathCategoryFeatureKey)) == null ? weight : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKey))) + weight)));
                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                batch.put(bytes(pathCategoryFeatureKeyValue), Longs.toByteArray((getDb().get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? weight : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKeyValue))) + weight)));
            }
            getDb().write(wo, batch);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the batch to avoid resource leaks.
            batch.close();
            ro.snapshot().close();
        }
    }

    @Override
    public IClassification classify(Map<String, String> features, final boolean explainData) throws ClassifyException {
        Map<String, Long> explanation = null;
        if (explainData) {
            explanation = new HashMap();
        }
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(getDb().getSnapshot());
        try {
            String pathGlobal = pathGlobal();
            long globalCount = (getDb().get(ro, bytes(pathGlobal)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathGlobal))));
            if (explainData) {
                explanation.put(pathGlobal, globalCount);
            }
            double[] likelyhood = new double[getCategories().length];
            double likelyhoodTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                String pathCategory = pathCategory(category);
                long categoryCount = (getDb().get(ro, bytes(pathCategory)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategory))));
                if (explainData) {
                    explanation.put(pathCategory, categoryCount);
                }
                double product = 1.0d;
                for (Map.Entry<String, String> feature : features.entrySet()) {
                    String pathFeatureKey = pathFeatureKey(feature.getKey());
                    long featureCount = (getDb().get(ro, bytes(pathFeatureKey)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKey))));
                    if (explainData) {
                        explanation.put(pathFeatureKey, featureCount);
                    }
                    if (featureCount > 0) {
                        String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                        long categoryFeatureCount = (getDb().get(ro, bytes(pathCategoryFeatureKey)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKey))));
                        if (explainData) {
                            explanation.put(pathCategoryFeatureKey, categoryFeatureCount);
                        }
                        String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                        long categoryFeatureValueCount = (getDb().get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKeyValue))));
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
            return new ClassificationImpl(features, likelihoodsToProbas(likelyhood, likelyhoodTot), explanation);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }
    }
}
