import jcuda.nvrtc.JNvrtc.*
import jcuda.nvrtc.nvrtcResult.NVRTC_SUCCESS

plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven(url = "https://packages.jetbrains.team/maven/p/skija/maven")
}

val lwjglVersion = "3.3.1"

val lwjglNatives = Pair(
    System.getProperty("os.name")!!,
    System.getProperty("os.arch")!!
).let { (name, arch) ->
    when {
        arrayOf("Linux", "FreeBSD", "SunOS", "Unit").any { name.startsWith(it) } ->
            if (arrayOf("arm", "aarch64").any { arch.startsWith(it) })
                "natives-linux${if (arch.contains("64") || arch.startsWith("armv8")) "-arm64" else "-arm32"}"
            else
                "natives-linux"

        arrayOf("Mac OS X", "Darwin").any { name.startsWith(it) } ->
            "natives-macos${if (arch.startsWith("aarch64")) "-arm64" else ""}"

        arrayOf("Windows").any { name.startsWith(it) } ->
            if (arch.contains("64"))
                "natives-windows${if (arch.startsWith("aarch64")) "-arm64" else ""}"
            else
                "natives-windows-x86"

        else -> throw Error("Unrecognized or unsupported platform. Please set \"lwjglNatives\" manually")
    }
}

val jcudaVersion = "11.7.0"

val jcudaoOsString =
    if (System.getProperty("java.vendor") == "The Android Project") "android" else System.getProperty("os.name")
        .toLowerCase().let {
            when {
                it.startsWith("windows") -> "windows"
                it.startsWith("mac os") -> "apple"
                it.startsWith("linux") -> "linux"
                it.startsWith("sun") -> "sun"
                else -> "unknown"
            }
        }
val jcudaArchString = System.getProperty("os.arch").toLowerCase().let {
    when {
        it == "i386" || it == "x86" || it == "i686" -> "x86"
        it.startsWith("amd64") || it.startsWith("x86_64") -> "x86_64"
        it.startsWith("arm64") -> "arm64"
        it.startsWith("arm") -> "arm"
        it == "ppc" || it == "powerpc" -> "ppc"
        it.startsWith("ppc") -> "ppc_64"
        it.startsWith("sparc") -> "sparc"
        it.startsWith("mips64") -> "mips64"
        it.startsWith("mips") -> "mips"
        it.contains("risc") -> "risc"
        else -> "unknown"
    }
}

val cudaClassifier = "$jcudaoOsString-$jcudaArchString"

dependencies {
    implementation("org.jetbrains:annotations:23.0.0")

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.0")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.9.0")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")
    testImplementation("me.tongfei:progressbar:0.9.3")

    // Log4j
    implementation("org.apache.logging.log4j:log4j-core:2.18.0")

    // Core module
    implementation(project(":core"))

    // CUDA
    implementation(group = "org.jcuda", name = "jcuda", version = jcudaVersion) {
        isTransitive = false
    }
    implementation(group = "org.jcuda", name = "jcublas", version = jcudaVersion) {
        isTransitive = false
    }
    implementation(group = "org.jcuda", name = "jcudnn", version = jcudaVersion) {
        isTransitive = false
    }
    implementation(group = "org.jcuda", name = "jcurand", version = jcudaVersion) {
        isTransitive = false
    }

    // CUDA natives
    implementation(group = "org.jcuda", name = "jcuda-natives", classifier = cudaClassifier, version = jcudaVersion)
    implementation(group = "org.jcuda", name = "jcublas-natives", classifier = cudaClassifier, version = jcudaVersion)
    implementation(group = "org.jcuda", name = "jcudnn-natives", classifier = cudaClassifier, version = jcudaVersion)
    implementation(group = "org.jcuda", name = "jcurand-natives", classifier = cudaClassifier, version = jcudaVersion)

    // LWJGL
    implementation(platform("org.lwjgl:lwjgl-bom:$lwjglVersion"))
    implementation("org.lwjgl", "lwjgl")
    implementation("org.lwjgl", "lwjgl-jemalloc")

    // LWJGL natives
    runtimeOnly("org.lwjgl", "lwjgl", classifier = lwjglNatives)
    runtimeOnly("org.lwjgl", "lwjgl-jemalloc", classifier = lwjglNatives)
}

// Also add CUDA to buildscript classpath for kernel compilation task
buildscript {
    val jcudaVersion = "11.7.0"

    val jcudaoOsString =
        if (System.getProperty("java.vendor") == "The Android Project") "android" else System.getProperty("os.name")
            .toLowerCase().let {
                when {
                    it.startsWith("windows") -> "windows"
                    it.startsWith("mac os") -> "apple"
                    it.startsWith("linux") -> "linux"
                    it.startsWith("sun") -> "sun"
                    else -> "unknown"
                }
            }
    val jcudaArchString = System.getProperty("os.arch").toLowerCase().let {
        when {
            it == "i386" || it == "x86" || it == "i686" -> "x86"
            it.startsWith("amd64") || it.startsWith("x86_64") -> "x86_64"
            it.startsWith("arm64") -> "arm64"
            it.startsWith("arm") -> "arm"
            it == "ppc" || it == "powerpc" -> "ppc"
            it.startsWith("ppc") -> "ppc_64"
            it.startsWith("sparc") -> "sparc"
            it.startsWith("mips64") -> "mips64"
            it.startsWith("mips") -> "mips"
            it.contains("risc") -> "risc"
            else -> "unknown"
        }
    }

    val cudaClassifier = "$jcudaoOsString-$jcudaArchString"
    dependencies {
        classpath(group = "org.jcuda", name = "jcuda", version = jcudaVersion) {
            isTransitive = false
        }
        classpath(group = "org.jcuda", name = "jcuda-natives", classifier = cudaClassifier, version = jcudaVersion)
    }
}

fun checkNVRTC(err: Int) {
    if (err != NVRTC_SUCCESS) {
        throw IllegalStateException(nvrtcGetErrorString(err))
    }
}

tasks.create("compileCudaKernels") {
    // Compiles all .cu files in src/main/cuda
    // and places the resulting .ptx files in build/cuda.

    // Get all .cu files
    val cuFiles = fileTree(project.projectDir.resolve("src/main/cuda")).matching {
        include("**/*.cu")
    }

    // Get all .cuh header files
    val headerFiles = fileTree(project.projectDir.resolve("src/main/cuda")).matching {
        include("**/*.cuh")
    }
    val headerSources = headerFiles.map { it.readText(Charsets.UTF_8) }.toTypedArray()
    val includeNames = headerFiles.map { it.name }.toTypedArray()

    // Create output directory
    val outputDir = project.buildDir.resolve("cuda")
    doLast("Create output directory") {
        outputDir.mkdirs()
    }

    // Compile each file
    cuFiles.forEach { cuFile ->
        doLast("Compile ${cuFile.name}") {
            val ptxFile = outputDir.resolve(cuFile.nameWithoutExtension + ".ptx")
            val program = jcuda.nvrtc.nvrtcProgram()
            val sourceCode = cuFile.readText(Charsets.UTF_8)
            checkNVRTC(nvrtcCreateProgram(program, sourceCode, cuFile.name, headerSources.size, headerSources, includeNames))
            val compileStatus = nvrtcCompileProgram(program, 0, null)
            run {
                // Get log
                val log = arrayOfNulls<String>(1)
                nvrtcGetProgramLog(program, log)
                if (compileStatus != NVRTC_SUCCESS) {
                    error("Compilation failed: ${log[0]}")
                } else if (!log[0].isNullOrBlank()) {
                    println("Compilation succeeded: ${log[0]}")
                }
            }
            // Get PTX
            run {
                val ptx = arrayOfNulls<String>(1)
                nvrtcGetPTX(program, ptx)
                ptxFile.writeText(ptx[0]!!)
            }
        }
    }
}

tasks.processResources {
    // Process resources depends on compileCudaKernels
    dependsOn("compileCudaKernels")

    // Copy all .ptx files from build/cuda into 'kernels/cuda'
    from(fileTree(project.buildDir.resolve("cuda")).matching {
        include("**/*.ptx")
    }) {
        into("kernels/cuda")
    }
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}