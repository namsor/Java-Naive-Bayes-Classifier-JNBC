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
 * Naive Bayes Classifier with Laplace smoothing and implementation with RocksDB
 * as key/value store. Learning is Synchronized but classification is not.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierRocksDBLaplacedImpl extends AbstractNaiveBayesClassifierRocksDBImpl implements INaiveBayesClassifier {

    private static final boolean VARIANT = false;
    private static final double ALPHA = 1d;
    private final boolean variant;
    private final double alpha;

    /**
     * Create a Naive Bayes Classifier implementation with RocksDB as key/value store.
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param rootPathWritable A writable path for RocksDB
     * @param alpha            The Laplace alpha, typically 1.0
     * @param variant          The Laplace variant
     * @throws PersistentClassifierException The persistence error and cause
     */

    public NaiveBayesClassifierRocksDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable, double alpha, boolean variant) throws PersistentClassifierException {
        super(classifierName, categories, rootPathWritable);
        this.variant = variant;
        this.alpha = alpha;
    }

    /**
     * Create a Naive Bayes Classifier implementation with RocksDB as key/value store and defaults for Laplace smoothing (ALPHA=1 and VARIANT=false)
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param rootPathWritable A writable path for RocksDB
     * @throws PersistentClassifierException The persistence error and cause
     */
    public NaiveBayesClassifierRocksDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable) throws PersistentClassifierException {
        this(classifierName, categories, rootPathWritable, ALPHA, VARIANT);
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
            String pathGlobalCountCategories = pathGlobalCountCategories();
            if (getDb().get(ro, bytes(pathCategory)) == null) {
                batch.put(bytes(pathCategory), Longs.toByteArray(weight));
                // increment the count
                batch.put(bytes(pathGlobalCountCategories), Longs.toByteArray((getDb().get(ro, bytes(pathGlobalCountCategories)) == null ? 1 : Longs.fromByteArray(getDb().get(ro, bytes(pathGlobalCountCategories))) + 1)));
            } else {
                batch.put(bytes(pathCategory), Longs.toByteArray(Longs.fromByteArray(getDb().get(ro, bytes(pathCategory))) + weight));
            }
            for (Map.Entry<String, String> feature : features.entrySet()) {
                String pathFeatureKey = pathFeatureKey(feature.getKey());
                batch.put(bytes(pathFeatureKey), Longs.toByteArray((getDb().get(ro, bytes(pathFeatureKey)) == null ? weight : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKey))) + weight)));
                String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                batch.put(bytes(pathCategoryFeatureKey), Longs.toByteArray((getDb().get(ro, bytes(pathCategoryFeatureKey)) == null ? weight : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKey))) + weight)));
                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                String pathFeatureKeyValue = pathFeatureKeyValue(feature.getKey(), feature.getValue());
                if (getDb().get(ro, bytes(pathFeatureKeyValue)) == null) {
                    batch.put(bytes(pathFeatureKeyValue), Longs.toByteArray(weight));
                    // increment the count
                    String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                    batch.put(bytes(pathFeatureKeyCountValueTypes), Longs.toByteArray((getDb().get(ro, bytes(pathFeatureKeyCountValueTypes)) == null ? 1 : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKeyCountValueTypes))) + 1)));
                } else {
                    batch.put(bytes(pathFeatureKeyValue), Longs.toByteArray(Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKeyValue))) + weight));
                }
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
            String pathGlobalCountCategories = pathGlobalCountCategories();
            long globalCount = (getDb().get(ro, bytes(pathGlobal)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathGlobal))));
            if (explainData) {
                explanation.put(pathGlobal, globalCount);
            }
            long globalCountCategories = (getDb().get(ro, bytes(pathGlobalCountCategories)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathGlobalCountCategories))));
            if (explainData) {
                explanation.put(pathGlobalCountCategories, globalCountCategories);
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
                        String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                        long featureCountValueTypes = (getDb().get(ro, bytes(pathFeatureKeyCountValueTypes)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKeyCountValueTypes))));
                        if (explainData) {
                            explanation.put(pathFeatureKeyCountValueTypes, featureCountValueTypes);
                        }
                        String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                        long categoryFeatureValueCount = (getDb().get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKeyValue))));
                        if (explainData) {
                            explanation.put(pathCategoryFeatureKeyValue, categoryFeatureValueCount);
                        }
                        double basicProbability = (categoryFeatureCount == 0 ? 0 : 1d * (categoryFeatureValueCount + alpha) / (categoryFeatureCount + featureCountValueTypes * alpha));
                        product *= basicProbability;
                    }
                }
                if (variant) {
                    likelyhood[i] = 1d * ((categoryCount + alpha) / (globalCount + globalCountCategories * alpha)) * product;
                } else {
                    likelyhood[i] = 1d * categoryCount / globalCount * product;
                }
                likelyhoodTot += likelyhood[i];
            }
            return new ClassificationImpl(features, likelihoodsToProbas(likelyhood, likelyhoodTot), explanation, true, variant, alpha, likelyhoodTot, false);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }
    }

}
