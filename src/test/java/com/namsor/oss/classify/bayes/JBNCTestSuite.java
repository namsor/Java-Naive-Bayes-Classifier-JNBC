/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
@Suite.SuiteClasses({com.namsor.oss.classify.bayes.NaiveBayesClassifierTransientImplV2Test.class, com.namsor.oss.classify.bayes.NaiveBayesClassifierTransientLaplacedImplV2Test.class})
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
