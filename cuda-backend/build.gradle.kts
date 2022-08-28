plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven(url = "https://packages.jetbrains.team/maven/p/skija/maven")
}

val jcudaVersion = "11.7.0"

val osString = if (System.getProperty("java.vendor") == "The Android Project") "android" else System.getProperty("os.name").toLowerCase().let {
    when {
        it.startsWith("windows") -> "windows"
        it.startsWith("mac os") -> "apple"
        it.startsWith("linux") -> "linux"
        it.startsWith("sun") -> "sun"
        else -> "unknown"
    }
}
val archString = System.getProperty("os.arch").toLowerCase().let {
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

dependencies {
    implementation("org.jetbrains:annotations:23.0.0")

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.0")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.9.0")

    implementation("org.apache.logging.log4j:log4j-core:2.18.0")

    // CUDA
    implementation(group="org.jcuda", name="jcuda", version=jcudaVersion) {
        isTransitive = false
    }
    implementation(group="org.jcuda", name="jcublas", version=jcudaVersion) {
        isTransitive = false
    }
    implementation(group="org.jcuda", name="jcudnn", version=jcudaVersion) {
        isTransitive = false
    }

    // CUDA natives
    val classifier = "$osString-$archString"
    implementation(group = "org.jcuda", name = "jcuda-natives", classifier = classifier, version = jcudaVersion)
    implementation(group = "org.jcuda", name = "jcublas-natives", classifier = classifier, version = jcudaVersion)
    implementation(group = "org.jcuda", name = "jcudnn-natives", classifier = classifier, version = jcudaVersion)

    implementation(project(":core"))
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}