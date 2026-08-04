pipeline {
    triggers { pollSCM('') }  // Run tests whenever a new commit is detected.
    agent { dockerfile {args '--gpus all'}} // Use the Dockerfile defined in the root Flash-X directory
    stages {

        //=============================//
    	// Set up submodules and amrex //
        //=============================//
    	stage('Prerequisites'){ steps{
	    sh 'mpicc -v'
	    sh 'nvidia-smi'
	    sh 'nvcc -V'
	    sh 'git submodule update --init'
	    sh 'pip3 list'
		sh 'echo HOME=$HOME'
		sh 'echo USER=$USER'
		sh 'echo LOGNAME=$LOGNAME'
		sh 'pwd'
		sh 'env | sort'
}}



	//=======//
	// Tests //
	//=======//
// 	stage('Only Vacuum oscillations'){ steps {
//                 // Convert the notebook to a Python script
//                 //sh 'jupyter nbconvert --to script tests/main_vac_osc.ipynb'
//                 // Run the converted Python script
//                 sh 'python tests/main_vac_osc.py'
// 				sh 'find . -name "*.pdf"'
// 				archiveArtifacts artifacts: '*.pdf', allowEmptyArchive: true
//             }
// } 

// 	stage('Rogerro(2021) full Hamiltonian'){ steps{
// 		//sh 'jupyter nbconvert --to script tests/main_Rog.ipynb'
// 		sh 'python tests/main_Rog.py'
//         sh 'find . -name "*.pdf"'
//         archiveArtifacts artifacts: '**/*.pdf'
// 		sh 'rm -rf datafiles plots'
//     } 
// }
// 	stage('Richers(2021) MF Homogenous QC_FFI'){ steps{
// 		//sh 'jupyter nbconvert --to script tests/Homogenous_FFI_Richers.ipynb'
// 		sh 'python tests/Homogenous_FFI_Richers.py'
//         sh 'find . -name "*.pdf"'
//         archiveArtifacts artifacts: '**/*.pdf'
// 		sh 'rm -rf datafiles plots'

//     } 
// }
// 	stage('Richers(2021) MF Inomogenous QC_FFI'){ steps{
// 		//sh 'jupyter nbconvert --to script tests/Inhomogenous_FFI_Richers.ipynb'
// 		sh 'python tests/Inhomogenous_FFI_Richers.py'
//         sh 'find . -name "*.pdf"'
//         archiveArtifacts artifacts: '**/*.pdf'
// 		sh 'rm -rf datafiles plots'
//     } 
}
	stage('Josh Homogenous depolarization noise'){ steps{
		//sh 'jupyter nbconvert --to script tests/Homo_Josh_noise_depolarization.ipynb'
		sh 'python tests/Homo_Josh_noise_depolarization.py'
		sh 'find . -name "*.pdf"'
		archiveArtifacts artifacts: '**/*.pdf'
		sh 'rm -rf datafiles plots'
    }
}
	stage('Observables unit test'){ steps{
		sh 'python tests/test_observables.py'
    }
}

}// stages{

    post {
        always {
	    cleanWs(
	        cleanWhenNotBuilt: true,
		deleteDirs: true,
		disableDeferredWipeout: false,
		notFailBuild: true,
		patterns: [[pattern: 'amrex', type: 'EXCLUDE']] ) // allow amrex to be cached
	}
    }

} // pipeline{
