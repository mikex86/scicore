plugins {
    id("java")
//    id("com.google.protobuf") version "0.8.17"
}

group = "org.example"
version = "1.0-SNAPSHOT"

buildscript {
    repositories {
        mavenCentral()
    }
//    dependencies {
//        classpath("com.google.protobuf:protobuf-gradle-plugin:0.8.17")
//    }
}

//sourceSets {
//    main {
//        // This is to get intellij to resolve imports properly in the Protobuf intellisense
//        // Don't actually place java source files in this directory, thank you.
//        java {
//            srcDirs("src/main/protobuf")
//        }
//        java {
//            srcDirs("src/generated/main/java")
//        }
//        proto {
//            srcDirs("src/main/protobuf")
//            include("**/*.proto3")
//        }
//    }
//}
//
//protobuf {
//    protobuf.generatedFilesBaseDir = "$projectDir/src/generated"
//}

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

dependencies {
    implementation("org.jetbrains:annotations:23.0.0")

    implementation(project(":matplotlib"))

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.0")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.9.0")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")

    testImplementation("me.tongfei:progressbar:0.9.5")
    testImplementation("com.google.code.gson:gson:2.9.1")

    implementation("org.apache.logging.log4j:log4j-core:2.19.0")

    // Protobuf
    implementation("com.google.protobuf:protobuf-java:3.21.8")
    runtimeOnly("com.google.protobuf:protobuf-java-util:3.21.8")

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

    testLogging {
        events("passed", "skipped", "failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        showExceptions = true
        showCauses = true
        showStackTraces = true
    }
}