package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathCategory;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathGlobal;

import java.io.*;

import java.util.Map;
import org.rocksdb.ReadOptions;
import org.rocksdb.RocksDBException;
import org.rocksdb.WriteBatch;
import org.rocksdb.WriteOptions;

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
    
    public NaiveBayesClassifierRocksDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable, int topN) throws IOException, PersistentClassifierException {
        this(classifierName, categories, rootPathWritable, ALPHA, VARIANT, topN);
    }
    public NaiveBayesClassifierRocksDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable) throws IOException, PersistentClassifierException {
        this(classifierName, categories, rootPathWritable, ALPHA, VARIANT);
    }
    
    public NaiveBayesClassifierRocksDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable, double alpha, boolean variant) throws IOException, PersistentClassifierException {
        super(classifierName, categories, rootPathWritable);
        this.variant = variant;
        this.alpha = alpha;
    }

    public NaiveBayesClassifierRocksDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable, double alpha, boolean variant, int topN) throws IOException, PersistentClassifierException {
        super(classifierName, categories, rootPathWritable, topN);
        this.variant = variant;
        this.alpha = alpha;
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
    public IClassification[] classify(Map<String, String> features) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(getDb().getSnapshot());
        try {
            String pathGlobal = pathGlobal();
            String pathGlobalCountCategories = pathGlobalCountCategories();
            long globalCount = (getDb().get(ro, bytes(pathGlobal)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathGlobal))));
            long globalCountCategories = (getDb().get(ro, bytes(pathGlobalCountCategories)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathGlobalCountCategories))));
            double[] likelyhood = new double[getCategories().length];
            double likelyhoodTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                String pathCategory = pathCategory(category);
                long categoryCount = (getDb().get(ro, bytes(pathCategory)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategory))));
                double product = 1.0d;
                for (Map.Entry<String, String> feature : features.entrySet()) {
                    String pathFeatureKey = pathFeatureKey(feature.getKey());
                    double featureCount = (getDb().get(ro, bytes(pathFeatureKey)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKey))));
                    if (featureCount > 0) {
                        String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                        double categoryFeatureCount = (getDb().get(ro, bytes(pathCategoryFeatureKey)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKey))));
                        String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                        double featureCountValueTypes = (getDb().get(ro, bytes(pathFeatureKeyCountValueTypes)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathFeatureKeyCountValueTypes))));
                        String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                        double categoryFeatureValueCount = (getDb().get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? 0 : Longs.fromByteArray(getDb().get(ro, bytes(pathCategoryFeatureKeyValue))));
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
            return likelihoodsToProbas(likelyhood, likelyhoodTot);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }
    }

}
