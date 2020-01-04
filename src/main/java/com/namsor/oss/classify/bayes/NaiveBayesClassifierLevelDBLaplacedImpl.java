package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import org.iq80.leveldb.ReadOptions;
import org.iq80.leveldb.WriteBatch;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Naive Bayes Classifier implementation with Laplace smoothing and LevelDB as
 * key/value store. Learning is Synchronized but classification is not.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierLevelDBLaplacedImpl extends AbstractNaiveBayesClassifierLevelDBImpl implements INaiveBayesClassifier {

    private static final boolean VARIANT = false;
    private static final double ALPHA = 1d;
    private final boolean variant;
    private final double alpha;

    /**
     * Create a Naive Bayes Classifier implementation with Laplace smoothing and LevelDB as key/value store.
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param rootPathWritable A writable path for LevelDB storage
     * @param alpha            The laplace Alpha, usually 1.0
     * @param variant          True for variant likelyhood[i] = 1d * ((categoryCount + alpha) / (globalCount + globalCountCategories * alpha)) * product;
     * @throws PersistentClassifierException The persistence error and cause
     */
    public NaiveBayesClassifierLevelDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable, double alpha, boolean variant) throws PersistentClassifierException {
        super(classifierName, categories, rootPathWritable);
        this.variant = variant;
        this.alpha = alpha;
    }

    /**
     * Create a Naive Bayes Classifier implementation with Laplace smoothing and LevelDB as key/value store, with defaults ALPHA=1 and VARIANT=false
     *
     * @param classifierName   The classifier name
     * @param categories       The classification categories
     * @param rootPathWritable A writable path for LevelDB storage
     * @throws PersistentClassifierException The persistence error and cause
     */
    public NaiveBayesClassifierLevelDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable) throws PersistentClassifierException {
        this(classifierName, categories, rootPathWritable, ALPHA, VARIANT);
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
            String pathGlobalCountCategories = pathGlobalCountCategories();
            if (getDb().get(bytes(pathCategory), ro) == null) {
                batch.put(bytes(pathCategory), Longs.toByteArray(weight));
                // increment the count
                batch.put(bytes(pathGlobalCountCategories), Longs.toByteArray((getDb().get(bytes(pathGlobalCountCategories), ro) == null ? 1 : Longs.fromByteArray(getDb().get(bytes(pathGlobalCountCategories), ro)) + 1)));
            } else {
                batch.put(bytes(pathCategory), Longs.toByteArray(Longs.fromByteArray(getDb().get(bytes(pathCategory), ro)) + weight));
            }
            for (Map.Entry<String, String> feature : features.entrySet()) {
                String pathFeatureKey = pathFeatureKey(feature.getKey());
                batch.put(bytes(pathFeatureKey), Longs.toByteArray((getDb().get(bytes(pathFeatureKey), ro) == null ? weight : Longs.fromByteArray(getDb().get(bytes(pathFeatureKey), ro)) + weight)));
                String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                batch.put(bytes(pathCategoryFeatureKey), Longs.toByteArray((getDb().get(bytes(pathCategoryFeatureKey), ro) == null ? weight : Longs.fromByteArray(getDb().get(bytes(pathCategoryFeatureKey), ro)) + weight)));
                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                String pathFeatureKeyValue = pathFeatureKeyValue(feature.getKey(), feature.getValue());
                if (getDb().get(bytes(pathFeatureKeyValue), ro) == null) {
                    batch.put(bytes(pathFeatureKeyValue), Longs.toByteArray(weight));
                    // increment the count
                    String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                    batch.put(bytes(pathFeatureKeyCountValueTypes), Longs.toByteArray((getDb().get(bytes(pathFeatureKeyCountValueTypes), ro) == null ? 1 : Longs.fromByteArray(getDb().get(bytes(pathFeatureKeyCountValueTypes), ro)) + 1)));
                } else {
                    batch.put(bytes(pathFeatureKeyValue), Longs.toByteArray(Longs.fromByteArray(getDb().get(bytes(pathFeatureKeyValue), ro)) + weight));
                }
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
            String pathGlobalCountCategories = pathGlobalCountCategories();
            long globalCount = (getDb().get(bytes(pathGlobal), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathGlobal), ro)));
            if (explainData) {
                explanation.put(pathGlobal, globalCount);
            }
            long globalCountCategories = (getDb().get(bytes(pathGlobalCountCategories), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathGlobalCountCategories), ro)));
            if (explainData) {
                explanation.put(pathGlobalCountCategories, globalCountCategories);
            }
            double[] likelyhood = new double[getCategories().length];
            double likelyhoodTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                String pathCategory = pathCategory(category);
                long categoryCount = (getDb().get(bytes(pathCategory), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathCategory), ro)));
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
                        String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                        long featureCountValueTypes = (getDb().get(bytes(pathFeatureKeyCountValueTypes), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathFeatureKeyCountValueTypes), ro)));
                        if (explainData) {
                            explanation.put(pathFeatureKeyCountValueTypes, featureCountValueTypes);
                        }
                        String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                        long categoryFeatureValueCount = (getDb().get(bytes(pathCategoryFeatureKeyValue), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathCategoryFeatureKeyValue), ro)));
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
            return new ClassificationImpl(features, likelihoodsToProbas(likelyhood, likelyhoodTot), explanation, true, variant, alpha);
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
