package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathCategory;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathGlobal;

import java.io.*;
import java.util.HashMap;

import java.util.Map;
import org.rocksdb.ReadOptions;
import org.rocksdb.RocksDBException;
import org.rocksdb.WriteBatch;
import org.rocksdb.WriteOptions;

/**
 * Naive Bayes Classifier implementation with RocksDB as key/value store.
 * Learning is Synchronized but classification is not.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierRocksDBImpl extends AbstractNaiveBayesClassifierRocksDBImpl implements INaiveBayesClassifier {

    public NaiveBayesClassifierRocksDBImpl(String classifierName, String[] categories, String rootPathWritable, int topN) throws IOException, PersistentClassifierException {
        super(classifierName, categories, rootPathWritable, topN);
    }

    public NaiveBayesClassifierRocksDBImpl(String classifierName, String[] categories, String rootPathWritable) throws IOException, PersistentClassifierException {
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
    public IClassification classify(Map<String, String> features, final boolean explain) throws ClassifyException {
        Map<String, Long> explanation = null;
        if (explain) {
            explanation = new HashMap();
        }
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(getDb().getSnapshot());
        try {
            String pathGlobal = pathGlobal();
            long globalCount = (getDb().get(ro, bytes(pathGlobal)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathGlobal))));
            if (explain) {
                explanation.put(pathGlobal, globalCount);
            }
            double[] likelyhood = new double[getCategories().length];
            double likelyhoodTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                String pathCategory = pathCategory(category);
                long categoryCount = (getDb().get(ro, bytes(pathCategory)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategory))));
                if (explain) {
                    explanation.put(pathCategory, categoryCount);
                }
                double product = 1.0d;
                for (Map.Entry<String, String> feature : features.entrySet()) {
                    String pathFeatureKey = pathFeatureKey(feature.getKey());
                    long featureCount = (getDb().get(ro, bytes(pathFeatureKey)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKey))));
                    if (explain) {
                        explanation.put(pathFeatureKey, featureCount);
                    }
                    if (featureCount > 0) {
                        String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                        long categoryFeatureCount = (getDb().get(ro, bytes(pathCategoryFeatureKey)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKey))));
                        if (explain) {
                            explanation.put(pathCategoryFeatureKey, categoryFeatureCount);
                        }
                        String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                        long categoryFeatureValueCount = (getDb().get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKeyValue))));
                        if (explain) {
                            explanation.put(pathCategoryFeatureKeyValue, categoryFeatureValueCount);
                        }
                        double basicProbability = (categoryFeatureCount == 0 ? 0 : 1d * categoryFeatureValueCount / categoryFeatureCount);
                        product *= basicProbability;
                    }
                }
                likelyhood[i] = 1d * categoryCount / globalCount * product;
                likelyhoodTot += likelyhood[i];
            }
            return new ClassificationImpl(likelihoodsToProbas(likelyhood, likelyhoodTot), explanation);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }

    }

}
