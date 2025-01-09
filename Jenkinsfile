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
}}



	//=======//
	// Tests //
	//=======//
	stage('Only Vacuum oscillations'){ steps {
                // Convert the notebook to a Python script
                sh 'jupyter nbconvert --to script tests/main_vac_osc.ipynb'
                // Run the converted Python script
                sh 'python tests/main_vac_osc.py'
				archiveArtifacts artifacts: '*.pdf'
            }
} 
	stage('Rogerro(2021)_only_self_interactions'){ steps{
		sh 'jupyter nbconvert --to script tests/main_self_int_Rog.ipynb'
		sh 'python tests/main_self_int_Rog.py'
		archiveArtifacts artifacts: '*.pdf'
    } 
}
	stage('Rogerro(2021) full Hamiltonian'){ steps{
		sh 'jupyter nbconvert --to script tests/main_Rog.ipynb'
		sh 'python tests/main_Rog.py'
		archiveArtifacts artifacts: '*.pdf'
    } 
}
	stage('Richers(2021) MF Homogenous QC_FFI'){ steps{
		sh 'jupyter nbconvert --to script tests/Homogenous_FFI_Richers.ipynb'
		sh 'python tests/Homogenous_FFI_Richers.py'
		archiveArtifacts artifacts: 'misc/plots/FFI/*/*/*/*.pdf'
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
