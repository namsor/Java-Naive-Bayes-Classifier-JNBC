package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathCategory;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathGlobal;

import java.io.*;

import java.util.Map;
import static org.fusesource.leveldbjni.JniDBFactory.asString;
import org.iq80.leveldb.DBIterator;
import org.iq80.leveldb.ReadOptions;
import org.iq80.leveldb.WriteBatch;

/**
 * Naive Bayes Classifier implementation with LevelDB as key/value store.
 * Learning is Synchronized but classification is not.
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierLevelDBImpl extends AbstractNaiveBayesClassifierLevelDBImpl implements INaiveBayesClassifier {

    public NaiveBayesClassifierLevelDBImpl(String classifierName, String[] categories, String rootPathWritable, int topN) throws IOException, PersistentClassifierException {
        super(classifierName, categories, rootPathWritable, topN);
    }

    public NaiveBayesClassifierLevelDBImpl(String classifierName, String[] categories, String rootPathWritable) throws IOException, PersistentClassifierException {
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
    public IClassification[] classify(Map<String, String> features) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.snapshot(getDb().getSnapshot());
        try {
            String pathGlobal = pathGlobal();
            long globalCount = (getDb().get(bytes(pathGlobal), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathGlobal), ro)));            
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
                    double featureCount = (getDb().get(bytes(pathFeatureKey), ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathFeatureKey),ro)));
                    if (featureCount > 0) {
                        String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                        double categoryFeatureCount = (getDb().get(bytes(pathCategoryFeatureKey),ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathCategoryFeatureKey),ro)));
                        String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                        double categoryFeatureValueCount = (getDb().get(bytes(pathCategoryFeatureKeyValue),ro) == null ? 0 : Longs.fromByteArray(getDb().get(bytes(pathCategoryFeatureKeyValue),ro)));
                        double basicProbability = (categoryFeatureCount == 0 ? 0 : 1d * categoryFeatureValueCount / categoryFeatureCount);
                        product *= basicProbability;
                    }
                }
                likelyhood[i] = 1d * categoryCount / globalCount * product;
                likelyhoodTot += likelyhood[i];
            }
            return likelihoodsToProbas(likelyhood, likelyhoodTot);
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
