/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.namsor.oss.classify.bayes;

import com.google.common.primitives.Longs;
import static com.namsor.oss.classify.bayes.AbstractNaiveBayesClassifierImpl.KEY_GLOBAL;
import java.io.*;
import java.util.Arrays;
import java.util.Set;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import org.lmdbjava.*;

import static java.nio.ByteBuffer.allocateDirect;
import static java.nio.charset.StandardCharsets.UTF_8;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.lmdbjava.CursorIterator.KeyVal;

import static org.lmdbjava.DbiFlags.MDB_CREATE;
import static org.lmdbjava.Env.create;

/**
 * Implementation : not working (limitation on KEY SIZE, would need recompile)
 *
 * @author elian carsenat, NamSor SAS
 */
public class NaiveBayesClassifierLMDBImpl extends AbstractNaiveBayesClassifierImpl implements INaiveBayesClassifier {

    private final String rootPathWritable;
    private final Env<ByteBuffer> env;
    private final Dbi<ByteBuffer> db;
    private final int maxKeySize;
    private final int maxValueSize;

    public NaiveBayesClassifierLMDBImpl(String classifierName, String[] categories, String rootPathWritable, long mapSize) throws IOException {
        super(classifierName, categories);
        this.rootPathWritable = rootPathWritable + "/lmdb_" + classifierName;
        final File path = new File(rootPathWritable);
        if (!path.exists()) {
            Logger.getLogger(getClass().getName()).info("Creating " + path.getAbsolutePath());
            path.mkdirs();
            Logger.getLogger(getClass().getName()).info(path.getAbsolutePath() + " created");
        }
        if (!path.canWrite()) {
            Logger.getLogger(getClass().getName()).info(path.getAbsolutePath() + " is not writable!");
        }
        env = create()
                // LMDB also needs to know how large our DB might be. Over-estimating is OK.
                .setMapSize(mapSize)
                // LMDB also needs to know how many DBs (Dbi) we want to store in this Env.
                .setMaxDbs(1)
                // Now let's open the Env. The same path can be concurrently opened and
                // used in different processes, but do not open the same path twice in
                // the same process at the same time.
                .open(path);
        maxKeySize = env.getMaxKeySize();
        Logger.getLogger(getClass().getName()).info("Created env, maxKeySize=" + maxKeySize);
        // We need a Dbi for each DB. A Dbi roughly equates to a sorted map. The
        // MDB_CREATE flag causes the DB to be created if it doesn't already exist.
        db = env.openDbi(classifierName, MDB_CREATE);
        maxValueSize = Longs.toByteArray(1l).length;
    }

    public String dbStatus() {
        return "OK";
    }

    public void dbClose() throws IOException {
    }

    public void dbCloseAndDestroy() throws IOException {
    }

    @Override
    public void learn(String category, Set<String> features, int weight) throws ClassifyException {
        // As per tutorial1...
        final ByteBuffer key = allocateDirect(maxKeySize);
        final ByteBuffer val = allocateDirect(maxValueSize);

        try (Txn<ByteBuffer> txn = env.txnWrite()) {
            // A cursor always belongs to a particular Dbi.
            final Cursor<ByteBuffer> c = db.openCursor(txn);
            key.put(KEY_GLOBAL.getBytes(UTF_8)).flip();
            val.put((db.get(txn, key) == null ? Longs.toByteArray(weight) : Longs.toByteArray(Longs.fromByteArray(db.get(txn, key).array()) + weight))).flip();
            key.put((KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category).getBytes(UTF_8)).flip();
            val.put((db.get(txn, key) == null ? Longs.toByteArray(weight) : Longs.toByteArray(Longs.fromByteArray(db.get(txn, key).array()) + weight))).flip();
            //db.put(bytes(KEY_GLOBAL), Longs.toByteArray((db.get(bytes(KEY_GLOBAL),ro) == null ? 1 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL),ro)) + weight)));
            //db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category),ro) == null ? 1 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category),ro)) + weight)));
            for (String feature : features) {
                key.put((KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature).getBytes(UTF_8)).flip();
                val.put((db.get(txn, key) == null ? Longs.toByteArray(weight) : Longs.toByteArray(Longs.fromByteArray(db.get(txn, key).array()) + weight))).flip();
                key.put((KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature).getBytes(UTF_8)).flip();
                val.put((db.get(txn, key) == null ? Longs.toByteArray(weight) : Longs.toByteArray(Longs.fromByteArray(db.get(txn, key).array()) + weight))).flip();
                //db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature),ro) == null ? 1 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature),ro)) + weight)));
                //db.put(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), Longs.toByteArray((db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature),ro) == null ? 1 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature),ro)) + weight)));
            }
            c.close();
            txn.commit();
        }
    }

    @Override
    public void forget(String category, Set<String> features) throws ClassifyException {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }


    @Override
    public IClassification[] classify(Set<String> features) throws ClassifyException {
        IClassification[] result = new ClassificationImpl[getCategories().length];
        try (Txn<ByteBuffer> txn = env.txnRead()) {
            // A cursor always belongs to a particular Dbi.
            txn.close();
        }
        try (Txn<ByteBuffer> rtx = env.txnRead()) {
            // Open a read-only Txn. It only sees data that existed at Txn creation time.
            final ByteBuffer key = allocateDirect(maxKeySize);
            //final ByteBuffer val = allocateDirect(maxValueSize);
            key.put(KEY_GLOBAL.getBytes(UTF_8)).flip();
            long globalCount = (db.get(rtx, key) == null ? 0 : Longs.fromByteArray(db.get(rtx, key).array()));
            for (int i = 0; i < getCategories().length; i++) {
                String category = getCategories()[i];
                key.put((KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category).getBytes(UTF_8)).flip();
                long categoryCount = (db.get(rtx, key) == null ? 0 : Longs.fromByteArray(db.get(rtx, key).array())); //(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), ro) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category), ro)));
                double product = 1.0d;
                double weight = 1.0d;
                double assumedProbability = 0.5d;
                for (String feature : features) {
                    //product *= this.featureWeighedAverage(feature, category); //this.featureWeighedAverage(feature, category, null, 1.0d, 0.5d);
                    key.put((KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature).getBytes(UTF_8)).flip();
                    double featureCount = (db.get(rtx, key) == null ? 0 : Longs.fromByteArray(db.get(rtx, key).array())); //(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro)));
                    key.put((KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature).getBytes(UTF_8)).flip();
                    double featureCategoryCount = (db.get(rtx, key) == null ? 0 : Longs.fromByteArray(db.get(rtx, key).array())); // (db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro) == null ? 0 : Longs.fromByteArray(db.get(bytes(KEY_GLOBAL + KEY_SEPARATOR + KEY_CATEGORY + KEY_SEPARATOR + category + KEY_SEPARATOR + KEY_FEATURE + KEY_SEPARATOR + feature), ro)));
                    double basicProbability = (featureCount == 0 ? 0 : 1d * featureCategoryCount / featureCount);

                    double featureWeighedAverage = (weight * assumedProbability + featureCount * basicProbability) / (weight + featureCount);
                    product *= featureWeighedAverage;
                }
                double proba = 1d * categoryCount / globalCount * product;
                IClassification classif = new ClassificationImpl(category, proba); // return ((double) this.getCategoryCount(category) / (double) this.getCategoriesTotal()) * featuresProbabilityProduct(features, category);
                result[i] = classif;
            }
        }
        Arrays.sort(result, orderByProba);
        return result;
    }

    public void dumpDb(Writer w) throws ClassifyException {
        try (Txn<ByteBuffer> rtx = env.txnRead()) {
            // Each iterator uses a cursor and must be closed when finished.
            // Iterate forward in terms of key ordering starting with the first key.
            try (CursorIterator<ByteBuffer> it = db.iterate(rtx, KeyRange.all())) {
                for (final KeyVal<ByteBuffer> kv : it.iterable()) {
                    String key = UTF_8.decode(kv.key()).toString();
                    long value = Longs.fromByteArray(kv.val().array());
                     w.append(key + "|" + value + "\n");
                }
            } catch (IOException ex) {
                Logger.getLogger(NaiveBayesClassifierLMDBImpl.class.getName()).log(Level.SEVERE, null, ex);
                throw new ClassifyException(ex);
            }
        }
    }

}
