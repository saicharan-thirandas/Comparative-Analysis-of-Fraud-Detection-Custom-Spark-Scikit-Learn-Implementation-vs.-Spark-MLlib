# Makefile for Spark WordCount project.

# Customize these paths for your environment.
# -----------------------------------------------------------
spark.root=/usr/local/spark-3.4.0-bin-without-hadoop
hadoop.root=/usr/local/hadoop-3.3.5

app.name=Spark-Base-Ensemble
spark-main-file = src/spark-base/run.py

#app.name=Spark-MLLib-Ensemble
#spark-main-file = src/spark-mllib/run.py




#local paths
local.master=local[4]
local.input=input
local.output=output
local.conf=conf
local.src=src



# Pseudo-Cluster Execution
hdfs.user.name=joe
hdfs.input=input
hdfs.output=output
# AWS EMR Execution
aws.emr.release=emr-6.10.0
aws.bucket.name=cs6240-project-bucket-saicharan-pyspark
aws.input=input
aws.output=output
aws.log.dir=log
aws.num.nodes=1
aws.conf=conf
aws.src=src
aws.instance.type=m5.xlarge
# -----------------------------------------------------------


# Create S3 bucket.
make-bucket:
	aws s3 mb s3://${aws.bucket.name}

# Upload data to S3 input dir.
upload-input-aws: make-bucket
	aws s3 sync ${local.input} s3://${aws.bucket.name}/${aws.input}

# Upload data to S3 input dir.
upload-conf-aws: make-bucket
	aws s3 sync ${local.conf} s3://${aws.bucket.name}/${aws.conf}

# Upload data to S3 input dir.
upload-src-aws: make-bucket
	aws s3 sync ${local.src} s3://${aws.bucket.name}/${aws.src}


# Delete S3 output dir.
delete-output-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.output}*"

# Upload application to S3 bucket.
upload-app-aws:
	aws s3 cp ${jar.name} s3://${aws.bucket.name}

# Main EMR launch.
aws: upload-src-aws upload-conf-aws
	aws emr create-cluster \
		--name ${app.name} \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--bootstrap-actions '[{"Path":"s3://${aws.bucket.name}/conf/install-python-dependencies.sh","Name":"Install Python Dependencies"}]' \
   		--steps Type=Spark,Name="MyPySparkStep",ActionOnFailure=CONTINUE,Args=[--deploy-mode,cluster,--master,yarn,s3://${aws.bucket.name}/${spark-main-file}] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate


# Download output from S3.
download-output-aws: clean-local-output
	mkdir ${local.output}
	aws s3 sync s3://${aws.bucket.name}/${aws.output} ${local.output}

# Change to standalone mode.
switch-standalone:
	cp config/standalone/*.xml ${hadoop.root}/etc/hadoop

# Change to pseudo-cluster mode.
switch-pseudo:
	cp config/pseudo/*.xml ${hadoop.root}/etc/hadoop

# Package for release.
distro:
	rm -f Spark-Demo.tar.gz
	rm -f Spark-Demo.zip
	rm -rf build
	mkdir -p build/deliv/Spark-Demo
	cp -r src build/deliv/Spark-Demo
	cp -r config build/deliv/Spark-Demo
	cp -r input build/deliv/Spark-Demo
	cp pom.xml build/deliv/Spark-Demo
	cp Makefile build/deliv/Spark-Demo
	cp README.txt build/deliv/Spark-Demo
	tar -czf Spark-Demo.tar.gz -C build/deliv Spark-Demo
	cd build/deliv && zip -rq ../../Spark-Demo.zip Spark-Demo
	