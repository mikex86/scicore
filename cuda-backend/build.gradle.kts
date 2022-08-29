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

val osString =
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

    // CUDA natives
    val classifier = "$osString-$archString"
    implementation(group = "org.jcuda", name = "jcuda-natives", classifier = classifier, version = jcudaVersion)
    implementation(group = "org.jcuda", name = "jcublas-natives", classifier = classifier, version = jcudaVersion)
    implementation(group = "org.jcuda", name = "jcudnn-natives", classifier = classifier, version = jcudaVersion)

    // LWJGL
    implementation(platform("org.lwjgl:lwjgl-bom:$lwjglVersion"))
    implementation("org.lwjgl", "lwjgl")
    implementation("org.lwjgl", "lwjgl-jemalloc")

    // LWJGL natives
    runtimeOnly("org.lwjgl", "lwjgl", classifier = lwjglNatives)
    runtimeOnly("org.lwjgl", "lwjgl-jemalloc", classifier = lwjglNatives)
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}