package com.namsor.oss.classify.bayes;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/**
 *
 * @author elian
 */
@RunWith(Suite.class)
@Suite.SuiteClasses({com.namsor.oss.classify.bayes.NaiveBayesClassifierTransientImplTest.class, 
                     com.namsor.oss.classify.bayes.NaiveBayesClassifierTransientLaplacedImplTest.class,
                     com.namsor.oss.classify.bayes.NaiveBayesClassifierRocksDBImplTest.class,
                     com.namsor.oss.classify.bayes.NaiveBayesClassifierRocksDBLaplacedImplTest.class,

})
public class JBNCTestSuite {

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
