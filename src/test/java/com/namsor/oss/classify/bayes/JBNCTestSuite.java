package com.namsor.oss.classify.bayes;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

//todo remove comment

/**
 * @author elian
 */
@RunWith(Suite.class)
@Suite.SuiteClasses({com.namsor.oss.classify.bayes.NaiveBayesClassifierMapImplTest.class,
        com.namsor.oss.classify.bayes.NaiveBayesClassifierMapLaplacedImplTest.class,
        com.namsor.oss.classify.bayes.NaiveBayesClassifierLevelDBImplTest.class,
        com.namsor.oss.classify.bayes.NaiveBayesClassifierLevelDBLaplacedImplTest.class,
        com.namsor.oss.classify.bayes.NaiveBayesClassifierRocksDBImplTest.class,
        com.namsor.oss.classify.bayes.NaiveBayesClassifierRocksDBLaplacedImplTest.class,

})
public class JBNCTestSuite {

    //todo Remove unused methods (all of them)
    @BeforeClass
    public static void setUpClass() throws Exception {
    }

    @AfterClass
    public static void tearDownClass() throws Exception {
    }

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

}
