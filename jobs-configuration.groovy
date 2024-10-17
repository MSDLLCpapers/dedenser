jobs {
    "jte-multibranch-job" {
        jobType = "MULTIBRANCH_JOB_JTE"
        jobDescription = '''Description'''
        jteConfig {
            configBaseDir = "cicd/pipeline-configuration"
        }
        jteLibSources {
            lib {
                repositoryUrl = "https://github.com/MSDLLCpapers/dedenser.git"
                branchName = "*/master"
                libraryBaseDir = "libraries"
            }
        }
    }
}
