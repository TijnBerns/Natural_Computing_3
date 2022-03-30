package language_classifier;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import algorithms.ContiguousCountingDAG;
import algorithms.PatternTrie;
import algorithms.RChunkPatterns;
import alphabets.Alphabet;
import alphabets.AminoAcidAlphabet;
import alphabets.BinaryAlphabet;
import alphabets.BinaryLetterAlphabet;
import alphabets.DegenerateAminoAcidAlphabet;
import alphabets.LatinAlphabet;
import be.cylab.java.roc.CurveCoordinates;
import be.cylab.java.roc.Main;
import be.cylab.java.roc.Roc;
import be.cylab.java.roc.Utils;
import util.Debug;
import util.Settings;


public class LanguageClassifier {
	// Parameters to use
	static int n = 10;	// Length of strings in self set
	static int r = 3; 	// Parameter r <= n
	static int r2 = 0; // Output k-th component of matching profile (0 for full profile)
	static String self = "lang\\english.train";	// File containing self set (1 string per line)
	static String alpha = null; 			// Alphabet, currently one of [infer|binary|binaryletter|amino|damino|latin]. Default: infer (uses all characters from \\\"self\\\" file as alphabet). Alternatively, specify file://[f] to set the alphabet to all characters found in file [f].
	static boolean invertmatch = false;	
	static boolean matching_profile = false;// true if r2 > 0
	static boolean logarithmize = true;		// Output logarithms instead of actual values
	static boolean counting = true;			// Count matching detectors instead of binary match
	
	
	public static double getMatchingDetectors(String line, int i1, int i2, PatternTrie matcher, ContiguousCountingDAG counter, long baseline) {
		if( line.length() < n ){
			System.out.print( "NaN" );
		}
		
		double nmatch = 0;
		
    	// r-contiguous detectors
    	long last_result = -1;
    	for( int i = i1 ; i <= i2 ; i ++ ){
    		if( last_result != 0 ){
    			if( counting ){
					last_result = counter.countStringsThatMatch(line,i) - (i<r?baseline:0);
				} else {
					int rm = 1;
					while( matcher.matches(line,rm) >= rm && rm <= n ){
						rm ++;
					}
						
					last_result = rm-1;
				}
             
    			if( logarithmize ){
    				nmatch += Math.log(1+last_result)/Math.log(2.);
    			} else {
    				nmatch += last_result;
    			}
    		}
    	} 
	     
	    return nmatch;
	}
	
	public static void negativeSelection(String second_lang) throws FileNotFoundException {
		Settings.DEBUG = false;	// Print debug information
		
		// Set the alphabet to all characters found in file 
		Alphabet.set(new BinaryAlphabet());
		if (!new File(self).canRead()) {
			throw new IllegalArgumentException("Can't read file " + self);
		}
		Alphabet.set(new Alphabet(new File(self)));
		
		if (r < 0 || n <= 0 || r > n) {
			throw new IllegalArgumentException(
					"Illegal value(s) for n and/or r");
		}
	
		Debug.log("constructing matcher");
		List<PatternTrie> chunks = RChunkPatterns.rChunkPatterns(self, n, r, 0);
		PatternTrie matcher = null;
		ContiguousCountingDAG counter = null;
		long baseline = 0;  
			
		if( counting ){
			counter = new ContiguousCountingDAG(chunks, n, r);
		} else {
			matcher = RChunkPatterns.rContiguousGraphWithFailureLinks(chunks, n, r);
		}
	
		// output matching lengths to be used
		int i1 = 0, i2 = 0;
		if( !matching_profile ){
			i1 = r; i2 = r; 
		} else {
			if( r2 > 0 && r2 <= n ){
				i1 = r2; i2 = r2;
			} else {
				i1 = 0; i2 = n;
			}
		}

		Debug.log("matcher constructed");
		
		// Scan english.test and generates scores
		File english_test = new File("lang\\english.test");
		Scanner scan = new Scanner(english_test);
		List<Double> score = new ArrayList<Double>();
		List<Double> true_alert = new ArrayList<Double>();
		while ( scan.hasNextLine() ) {
			String line = scan.nextLine().trim();
			score.add(getMatchingDetectors(line, i1, i2, matcher, counter, baseline));
			true_alert.add(0.0);
		}
		scan.close();
		
		// Scan second language file and generate scores
		File foreign_test = new File(second_lang);
		scan = new Scanner(foreign_test);
		while ( scan.hasNextLine() ) {
			String line = scan.nextLine().trim();
			score.add(getMatchingDetectors(line, i1, i2, matcher, counter, baseline));
			true_alert.add(1.0);
		}
		scan.close();
		
		// normalize scores and map to double array
		double max_score = Collections.max(score);
		double[] score_array = score.stream().mapToDouble(d -> d / max_score).toArray();
		double[] true_alert_array = true_alert.stream().mapToDouble(d -> d).toArray();
		
		// Generate ROC and compute AUC
		Roc roc = new Roc(score_array, true_alert_array);
		System.out.println(roc.computeAUC());

		roc.computeRocPointsAndGenerateCurve("ROC-curve.png");
	}
	
	
	public static void main(String[] args) {
		try {
			String second_lang = "lang\\xhosa.txt";
			negativeSelection(second_lang);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
