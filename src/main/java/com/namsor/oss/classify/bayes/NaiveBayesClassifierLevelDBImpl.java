/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import org.iq80.leveldb.*;

import java.io.*;

//import static org.iq80.leveldb.impl.Iq80DBFactory.*;
import static org.fusesource.leveldbjni.JniDBFactory.*;
import java.util.Arrays;
import java.util.Set;

/**
 * Implementation : choose one of 
 * org.iq80.leveldb.impl.Iq80DBFactory.*; //import static
 * org.fusesource.leveldbjni.JniDBFactory.*; //import static
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierLevelDBImpl extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    private final String rootPathWritable;
    private final DB db;

    public NaiveBayesClassifierLevelDBImpl(String classifierName, String[] categories, String rootPathWritable, int cacheSizeMb) throws IOException {
        super(classifierName, categories);
        this.rootPathWritable = rootPathWritable;        
        Options options = new Options();
        options.createIfMissing(true);
        options.cacheSize(cacheSizeMb * 1048576); // 100MB cache
        options.compressionType(CompressionType.SNAPPY);
        db = factory.open(new File(rootPathWritable + "/" + classifierName), options);
    }

    public String dbStatus() {
        return db.getProperty("leveldb.stats");
    }

    public void dbClose() throws IOException {
        db.close();
    }

    public void dbCloseAndDestroy() throws IOException {
        db.close();
        Options options = new Options();
        factory.destroy(new File(rootPathWritable + "/" + getClassifierName()), options);
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
    public synchronized void learn(String category, Set<String> features, int weight) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.snapshot(db.getSnapshot());
        WriteBatch batch = db.createWriteBatch();
        try {

            //private Map<K, Map<T, Counter>> featureCountPerCategory;
            db.put(bytes(KEY_GLOBAL), Longs.toByteArray((db.get(bytes(KEY_GLOBAL), ro) == null ? weight : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL), ro)) + weight)));
            db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), ro) == null ? weight : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), ro)) + weight)));
            for (String feature : features) {
                db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro) == null ? weight : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro)) + weight)));
                db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro) == null ? weight : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro)) + weight)));
            }
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
    public void forget(String category, Set<String> features) throws ClassifyException {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public IClassification[] classify(Set<String> features) throws ClassifyException {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        ReadOptions ro = new ReadOptions();
        ro.snapshot(db.getSnapshot());
        //int gets = 0;
        long sM = System.currentTimeMillis();
        try {
            // return ((double) this.getCategoryCount(category) / (double) this.getCategoriesTotal()) * featuresProbabilityProduct(features, category);
            long globalCount = (db.get(bytes(KEY_GLOBAL), ro) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL), ro)));
            //gets++;
            long categoryCountTot = 0;
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                long categoryCount = (db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), ro) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), ro)));
                //gets++;
                double product = 1.0d;
                double weight = 1.0d;
                double assumedProbability = 0.5d;
                for (String feature : features) {
                    //product *= this.featureWeighedAverage(feature, category); //this.featureWeighedAverage(feature, category, null, 1.0d, 0.5d);
                    long featureCount = (db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro)));
                    //gets++;
                    long featureCategoryCount = 0;
                    // optim
                    if (featureCount > 0) {
                        featureCategoryCount = (db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro)));
                        //gets++;
                    }
                    double basicProbability = (featureCount == 0 ? 0 : 1d * featureCategoryCount / featureCount);

                    double featureWeighedAverage = (weight * assumedProbability + featureCount * basicProbability) / (weight + featureCount);
                    product *= featureWeighedAverage;
                }
                double proba = 1d * categoryCount / globalCount * product;
                IClassification classif = new ClassificationImpl(category, proba); // return ((double) this.getCategoryCount(category) / (double) this.getCategoriesTotal()) * featuresProbabilityProduct(features, category);
                result[i] = classif;
                categoryCountTot += categoryCount;
            }
            if (globalCount == categoryCountTot) {
                // ok we're consistent
            } else {
                // there is a slight inconsistency bw/ globalCount and categoryCount ... should not be, but not a big deal
                // throw new ClassifyException("Inconsistency : globalcount="+globalCount+" categoryCountTot="+categoryCountTot);
            }
        } finally {
            try {
                // Make sure you close the snapshot to avoid resource leaks.
                ro.snapshot().close();
            } catch (IOException ex) {
                throw new ClassifyException(ex);
            }
        }
        //long fM = System.currentTimeMillis();
        //Logger.getLogger(getClass().getName()).info(getClassifierName() + " cats=" + this.getCategories().length + " features=" + features.size() + " did " + gets + " get in (" + (fM - sM) + " millis.");
        Arrays.sort(result, orderByProba);
        return result;
    }

    public void dumpDb(Writer w) throws ClassifyException {
        ReadOptions ro = new ReadOptions();
        ro.snapshot(db.getSnapshot());
        DBIterator iterator = db.iterator(ro);
        try {
            for (iterator.seekToFirst(); iterator.hasNext(); iterator.next()) {
                String key = asString(iterator.peekNext().getKey());
                long value = Longs.fromByteArray(iterator.peekNext().getValue());
                w.append(key + "|" + value + "\n");
            }
        } catch (IOException ex) {
            throw new ClassifyException(ex);
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
