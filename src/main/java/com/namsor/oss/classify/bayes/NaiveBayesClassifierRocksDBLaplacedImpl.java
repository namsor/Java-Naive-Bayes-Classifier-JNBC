package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathCategory;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.pathGlobal;
import org.rocksdb.RocksDB;
import org.rocksdb.Options;

import java.io.*;

import java.util.Arrays;
import java.util.Map;
import org.rocksdb.CompressionType;
import org.rocksdb.ReadOptions;
import org.rocksdb.RocksDBException;
import org.rocksdb.RocksIterator;
import org.rocksdb.WriteBatch;
import org.rocksdb.WriteOptions;

/**
 * Naive Bayes Classifier with Laplace smoothing and implementation with RocksDB as key/value store. 
 * Learning is Synchronized by classification is not.
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierRocksDBLaplacedImpl extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    private static final boolean VARIANT = false;
    private static final double ALPHA = 1d;
    private final boolean variant;
    private final double alpha;
    private final String rootPathWritable;
    private final RocksDB db;

    public NaiveBayesClassifierRocksDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable) throws IOException, PersistentClassifierException {
        this(classifierName, categories, rootPathWritable, ALPHA, VARIANT);
    }

    public NaiveBayesClassifierRocksDBLaplacedImpl(String classifierName, String[] categories, String rootPathWritable, double alpha, boolean variant) throws IOException, PersistentClassifierException {
        super(classifierName, categories);
        this.rootPathWritable = rootPathWritable;
        this.variant = variant;
        this.alpha = alpha;
        Options options = new Options();
        options.setCreateIfMissing(true);
        options.setCompressionType(CompressionType.NO_COMPRESSION);
        try {
            db = RocksDB.open(options, rootPathWritable + "/" + classifierName);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    public String dbStatus() throws PersistentClassifierException {
        try {
            return db.getProperty("leveldb.stats");
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    @Override
    public void dbClose() throws PersistentClassifierException {
        db.close();
    }

    @Override
    public void dbCloseAndDestroy() throws PersistentClassifierException {
        try {
            db.close();
            Options options = new Options();
            RocksDB.destroyDB(rootPathWritable + "/" + getClassifierName(), options);
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        }
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        try {
            dbClose();
        } catch (Throwable t) {
            // ignore
        }
    }

    @Override
    public synchronized void learn(String category, Map<String, String> features, long weight) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(db.getSnapshot());
        WriteOptions wo = new WriteOptions();
        WriteBatch batch = new WriteBatch();
        try {
            String pathGlobal = pathGlobal();
            batch.put(bytes(pathGlobal), Longs.toByteArray((db.get(ro, bytes(pathGlobal)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(pathGlobal))) + weight)));
            String pathCategory = pathCategory(category);
            String pathGlobalCountCategories = pathGlobalCountCategories();
            if (db.get(ro, bytes(pathCategory)) == null) {
                batch.put(bytes(pathCategory), Longs.toByteArray(weight));
                // increment the count
                batch.put(bytes(pathGlobalCountCategories), Longs.toByteArray((db.get(ro, bytes(pathGlobalCountCategories)) == null ? 1 : Longs.fromByteArray(db.get(ro, bytes(pathGlobalCountCategories))) + 1)));
            } else {
                batch.put(bytes(pathCategory), Longs.toByteArray(Longs.fromByteArray(db.get(ro, bytes(pathCategory))) + weight));
            }
            for (Map.Entry<String, String> feature : features.entrySet()) {
                String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                batch.put(bytes(pathCategoryFeatureKey), Longs.toByteArray((db.get(ro, bytes(pathCategoryFeatureKey)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(pathCategoryFeatureKey))) + weight)));
                String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                String pathFeatureKeyValue = pathFeatureKeyValue(feature.getKey(), feature.getValue());
                if (db.get(ro, bytes(pathFeatureKeyValue)) == null) {
                    batch.put(bytes(pathFeatureKeyValue), Longs.toByteArray(weight));
                    // increment the count
                    String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                    batch.put(bytes(pathFeatureKeyCountValueTypes), Longs.toByteArray((db.get(ro, bytes(pathFeatureKeyCountValueTypes)) == null ? 1 : Longs.fromByteArray(db.get(ro, bytes(pathFeatureKeyCountValueTypes))) + 1)));
                } else {
                    batch.put(bytes(pathFeatureKeyValue), Longs.toByteArray(Longs.fromByteArray(db.get(ro, bytes(pathFeatureKeyValue))) + weight));
                }
                batch.put(bytes(pathCategoryFeatureKeyValue), Longs.toByteArray((db.get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? weight : Longs.fromByteArray(db.get(ro, bytes(pathCategoryFeatureKeyValue))) + weight)));
            }
            db.write(wo, batch);
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
        IClassification[] result = new ClassificationImpl[getCategories().length];
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(db.getSnapshot());
        try {
            String pathGlobal = pathGlobal();
            String pathGlobalCountCategories = pathGlobalCountCategories();
            long globalCount = (db.get(ro, bytes(pathGlobal)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathGlobal))));
            long globalCountCategories = (db.get(ro, bytes(pathGlobalCountCategories)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathGlobalCountCategories))));
            double[] likelyhood = new double[getCategories().length];
            double likelyhoodTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                String pathCategory = pathCategory(category);
                long categoryCount = (db.get(ro, bytes(pathCategory)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathCategory))));
                double product = 1.0d;
                for (Map.Entry<String, String> feature : features.entrySet()) {
                    String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                    double featureCount = (db.get(ro, bytes(pathCategoryFeatureKey)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathCategoryFeatureKey))));
                    String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                    double featureCountValueTypes = (db.get(ro, bytes(pathFeatureKeyCountValueTypes)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathFeatureKeyCountValueTypes))));
                    String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                    double featureCategoryCount = (db.get(ro, bytes(pathCategoryFeatureKeyValue)) == null ? 0 : Longs.fromByteArray(db.get(ro, bytes(pathCategoryFeatureKeyValue))));
                    double basicProbability = (featureCount == 0 ? 0 : 1d * (featureCategoryCount + alpha) / (featureCount + featureCountValueTypes * alpha));
                    product *= basicProbability;
                }
                if (variant) {
                    likelyhood[i] = 1d * ((categoryCount + alpha) / (globalCount + globalCountCategories * alpha)) * product;
                } else {
                    likelyhood[i] = 1d * categoryCount / globalCount * product;
                }
                likelyhoodTot += likelyhood[i];
            }
            for (int i = 0; i < getCategories().length; i++) {
                double proba = likelyhood[i] / likelyhoodTot;
                ClassificationImpl classif = new ClassificationImpl(getCategories()[i], proba);
                result[i] = classif;
            }
            Arrays.sort(result, orderByProba);
            return result;
        } catch (RocksDBException ex) {
            throw new PersistentClassifierException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }
    }

    public void dumpDb(Writer w) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.setSnapshot(db.getSnapshot());

        RocksIterator iterator = db.newIterator(ro);
        try {
            for (iterator.seekToFirst(); iterator.isValid(); iterator.next()) {
                String key = new String(iterator.key());
                long value = Longs.fromByteArray(iterator.value());
                w.append(key + "|" + value + "\n");
            }
        } catch (IOException ex) {
            throw new ClassifyException(ex);
        } finally {
            // Make sure you close the snapshot to avoid resource leaks.
            ro.snapshot().close();
        }
    }

    private byte[] bytes(String key) {
        return key.getBytes();
    }
}
