from python import Python, PythonObject

# Configuration initialization
fn init_config(args: PythonObject, logger: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """Initialize and validate configuration"""
    # Validate synthetic data path
    if args.use_synthetic_data and not args.synthetic_data_path:
        var error_msg = "synthetic_data_path must be provided when using synthetic data."
        logger.error(error_msg)
        raise Error(error_msg)
    
    # Validate splits
    var total_split = args.train_split + args.val_split + args.test_split
    if Python.evaluate("abs")(total_split - 1.0) >= 1e-6:
        var error_msg = "Train, validation, and test splits must sum to 1.0"
        logger.error(error_msg)
        raise Error(error_msg)
    
    # Initialize configurations based on optuna or default values
    var model_config: PythonObject
    var training_config: PythonObject
    
    if args.use_optuna:
        model_config, training_config = init_optuna_config(args, logger)
    else:
        model_config, training_config = init_default_config(args, logger)
    
    return model_config, training_config

fn setup_trainer(args: PythonObject, logger: PythonObject) raises -> PythonObject:
    """Set up PyTorch Lightning trainer with appropriate callbacks"""
    var callbacks: List[PythonObject] = []
    
    # Initialize GrokfastCallback if enabled
    if args.use_grokfast:
        var GrokfastCallback = Python.import_module("gpt2_arc.src.utils").GrokfastCallback
        var grokfast_callback = GrokfastCallback(
            filter_type=args.grokfast_type,
            alpha=args.grokfast_alpha,
            lamb=args.grokfast_lamb,
            window_size=(args.grokfast_window_size if args.grokfast_type == "ma" else 100),
            warmup=True,
            trigger=False
        )
        callbacks.append(grokfast_callback)
        logger.info("GrokfastCallback added to the training callbacks.")
    
    # Add checkpointing callback
    if not args.no_checkpointing:
        var ModelCheckpoint = Python.import_module("pytorch_lightning.callbacks").ModelCheckpoint
        var checkpoint_filename = ("resume-" if args.model_checkpoint else "") + "checkpoint-step_{step}-val_loss_{val_loss:.4f}"
        var checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=checkpoint_filename,
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        )
        callbacks.append(checkpoint_callback)
        
        var ModelConfigSaver = create_model_config_saver(config)
        callbacks.append(ModelConfigSaver)
    
    # Initialize TensorBoard logger
    var tb_logger = None
    if not args.no_logging:
        var TensorBoardLogger = Python.import_module("pytorch_lightning.loggers").TensorBoardLogger
        tb_logger = TensorBoardLogger(
            save_dir="runs",
            name="experiment_" + String(results_collector.experiment_id)
        )
    
    # Initialize trainer
    var pytorch_lightning = Python.import_module("pytorch_lightning")
    var trainer = pytorch_lightning.Trainer(
        max_epochs=args.max_epochs,
        logger=tb_logger,
        callbacks=callbacks if callbacks else None,
        enable_checkpointing=not args.no_checkpointing,
        enable_progress_bar=not args.no_progress_bar,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=1.0,
        precision=16,
        accelerator=determine_accelerator(args),
        devices=determine_devices(args),
        strategy=determine_strategy(args),
        profiler=setup_profiler(args) if args.use_profiler else None,
        val_check_interval=args.val_check_interval
    )
    
    return trainer

fn load_datasets(args: PythonObject, config: PythonObject, logger: PythonObject) raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    """Load and prepare datasets"""
    logger.info("Loading datasets sequentially to avoid memory allocation issues")
    
    try:
        var train_data = load_dataset(args, config, dataset_type="train", all_synthetic_data=all_synthetic_data)
        var val_data = load_dataset(args, config, dataset_type="val")
        var test_data = load_dataset(args, config, dataset_type="test")
        
        logger.info("Training dataset source: " + ("synthetic data" if args.use_synthetic_data else "official ARC data"))
        logger.info("Validation dataset source: official ARC data")
        logger.info("Test dataset source: official ARC data")
        
        return train_data, val_data, test_data
    except Error as e:
        logger.error("Error loading datasets: " + String(e))
        raise

fn train_and_evaluate(
    trainer: PythonObject,
    model: PythonObject,
    train_data: PythonObject,
    val_data: PythonObject,
    test_data: PythonObject,
    args: PythonObject,
    logger: PythonObject
) raises -> Dict[String, Float64]:
    """Train the model and evaluate results"""
    try:
        # Train model
        logger.info("Starting model training")
        trainer.fit(model, train_data, val_data)
        
        # Run evaluation
        logger.info("Starting model evaluation on test dataset")
        var test_results = trainer.test(model=model, dataloaders=test_data)
        
        if test_results:
            return process_test_results(test_results, logger)
        else:
            logger.error("No test results available")
            raise Error("No test results available")
            
    except Error as e:
        handle_training_error(e, logger)
        raise

fn save_model_and_results(
    model: PythonObject,
    results: Dict[String, Float64],
    experiment_id: String,
    logger: PythonObject
) raises -> None:
    """Save the trained model and results"""
    var os_module = Python.import_module("os")
    
    # Save model
    var model_path = "final_model_" + experiment_id + ".pth"
    os_module.makedirs("checkpoints", exist_ok=True)
    
    var torch = Python.import_module("torch")
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": model.config.model.__dict__,
        "training_config": model.config.training.__dict__,
        "pad_symbol_idx": model.config.training.pad_symbol_idx,
        "symbol_freq": model.config.training.symbol_freq
    }, model_path)
    
    # Save results
    os_module.makedirs("results", exist_ok=True)
    var results_path = "results/experiment_" + experiment_id + ".json"
    save_results_to_json(results, results_path)
    
    logger.debug("Model saved to: " + model_path)
    logger.debug("Results saved to: " + results_path)


fn init_optuna_config(args: PythonObject, logger: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """Initialize configuration using Optuna for hyperparameter optimization"""
    var optuna = Python.import_module("optuna")
    logger.info("Loading best hyperparameters from Optuna study")
    
    # Get study name
    var study_name = args.optuna_study_name
    if study_name is None:
        study_name = get_default_study_name(args, optuna, logger)
    
    # Load or create study
    try:
        var study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)
    except KeyError:
        var study = optuna.create_study(study_name=study_name, storage=args.optuna_storage)
    
    var best_params = study.best_params
    logger.debug("Loaded best parameters: " + String(best_params))
    
    # Calculate derived parameters
    var n_head = 2 ** best_params["n_head_exp"]
    var n_embd = calculate_embedding_dim(n_head, best_params["n_embd_multiplier"])
    
    # Create configurations
    var ModelConfig = Python.import_module("gpt2_arc.src.config").ModelConfig
    var TrainingConfig = Python.import_module("gpt2_arc.src.config").TrainingConfig
    
    var model_config = ModelConfig(
        n_embd=n_embd,
        n_head=n_head,
        n_layer=best_params["n_layer"],
        dropout=best_params["dropout"],
        num_workers=args.num_workers if args.num_workers is not None else Python.import_module("multiprocessing").cpu_count(),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        pin_memory=not args.no_pin_memory,
    )
    
    var training_config = create_training_config(args, best_params)
    
    return model_config, training_config

fn init_default_config(args: PythonObject, logger: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """Initialize configuration using default or provided parameters"""
    var ModelConfig = Python.import_module("gpt2_arc.src.config").ModelConfig
    var TrainingConfig = Python.import_module("gpt2_arc.src.config").TrainingConfig
    
    var model_config = ModelConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        mamba_ratio=args.mamba_ratio,
        d_state=args.d_state,
        d_conv=args.d_conv,
        mamba_depth=args.mamba_depth,
        mamba_expand=args.mamba_expand,
    )
    
    var training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        use_gpu=args.use_gpu,
        log_level=args.log_level,
        use_synthetic_data=args.use_synthetic_data,
        synthetic_data_path=args.synthetic_data_path,
        use_grokfast=args.use_grokfast,
        grokfast_type=args.grokfast_type,
        grokfast_alpha=args.grokfast_alpha,
        grokfast_lamb=args.grokfast_lamb,
        grokfast_window_size=args.grokfast_window_size,
        include_pad_in_loss=args.include_pad_in_loss,
        include_pad_in_accuracy=args.include_pad_in_accuracy,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        pin_memory=args.pin_memory,
    )
    
    return model_config, training_config

fn calculate_symbol_frequencies(
    args: PythonObject,
    train_data: PythonObject,
    config: PythonObject,
    logger: PythonObject
) raises -> Dict[Int, Float64]:
    """Calculate symbol frequencies for balanced training"""
    if not args.enable_symbol_freq:
        logger.debug("Symbol frequency calculation is disabled. Using empty symbol_freq_dict.")
        return {}
    
    logger.debug("Calculating symbol frequencies as it is enabled.")
    var symbol_freq = train_data.get_symbol_frequencies()
    
    var symbol_freq_dict = {}
    for (i, freq) in symbol_freq.enumerate():
        symbol_freq_dict[i] = Float64(freq)
    
    var pad_symbol_idx = config.training.pad_symbol_idx
    symbol_freq_dict.pop(pad_symbol_idx, None)
    
    logger.debug("Removed pad_symbol_idx (" + String(pad_symbol_idx) + 
                ") from symbol_freq_dict. New length: " + String(len(symbol_freq_dict)))
    
    assert(len(symbol_freq_dict) == config.training.num_classes - 1,
           "Length of symbol_freq_dict (" + String(len(symbol_freq_dict)) + 
           ") does not match num_classes minus padding (" + 
           String(config.training.num_classes - 1) + ").")
    
    return symbol_freq_dict

fn setup_profiler(args: PythonObject) raises -> PythonObject:
    """Setup profiler if enabled"""
    if not args.use_profiler:
        return None
        
    var ProfilerActivity = Python.import_module("torch.profiler").ProfilerActivity
    var profiler = Python.import_module("lightning.pytorch.profilers").PyTorchProfiler(
        dirpath=args.profiler_dirpath,
        filename=args.profiler_filename,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    )
    return profiler

fn determine_accelerator(args: PythonObject) -> Tuple[String, Int, String]:
    """Determine accelerator, devices, and strategy based on args"""
    if args.accelerator == "tpu":
        return "tpu", "xla:1", "tpu_spawn"
    elif args.accelerator == "gpu":
        var torch = Python.import_module("torch")
        if torch.cuda.is_available():
            return "gpu", 1, "auto"
        return "cpu", 1, "auto"
    else:
        return "cpu", 1, "auto"

fn initialize_model(
    config: PythonObject,
    checkpoint_path: String,
    logger: PythonObject
) raises -> PythonObject:
    """Initialize or load model from checkpoint"""
    var torch = Python.import_module("torch")
    var GPT2ARC = Python.import_module("gpt2_arc.src.models.gpt2").GPT2ARC
    
    if checkpoint_path:
        logger.info("Loading model from checkpoint: " + checkpoint_path)
        var checkpoint = torch.load(checkpoint_path)
        validate_checkpoint(checkpoint, logger)
        return load_model_from_checkpoint(checkpoint, config, logger)
    
    logger.info("Initializing new model")
    var model = GPT2ARC(
        config=config,
        num_classes=config.training.num_classes,
        symbol_freq=config.training.symbol_freq,
        pad_symbol_idx=config.training.pad_symbol_idx
    )
    
    # Calculate layer distribution
    var mamba_layers = Int(config.model.n_layer * config.model.mamba_ratio)
    var transformer_layers = config.model.n_layer - mamba_layers
    validate_layer_distribution(mamba_layers, transformer_layers, config, logger)
    
    return model

fn process_test_results(
    test_results: List[Dict[String, Float64]], 
    logger: PythonObject
) -> Dict[String, Float64]:
    """Process and aggregate test results"""
    var sum_test_loss = 0.0
    var sum_test_accuracy = 0.0
    var sum_test_diff_accuracy = 0.0
    
    for result in test_results:
        sum_test_loss += result["avg_test_loss"]
        sum_test_accuracy += result["avg_test_accuracy"]
        sum_test_diff_accuracy += result["avg_test_diff_accuracy"]
    
    var avg_test_loss = sum_test_loss / Float64(len(test_results))
    var avg_test_accuracy = sum_test_accuracy / Float64(len(test_results))
    var avg_test_diff_accuracy = sum_test_diff_accuracy / Float64(len(test_results))
    
    logger.info("Test results - Loss: " + String(avg_test_loss) + 
                ", Accuracy: " + String(avg_test_accuracy) + 
                ", Diff Accuracy: " + String(avg_test_diff_accuracy))
    
    # Compile results dictionary
    var results = {
        "avg_test_loss": avg_test_loss,
        "avg_test_accuracy": avg_test_accuracy,
        "avg_test_diff_accuracy": avg_test_diff_accuracy
    }
    
    # Add task-specific results
    for result in test_results:
        for key, value in result.items():
            if key.endswith("_test_accuracy") or key.endswith("_test_diff_accuracy"):
                results[key] = value
    
    return results

fn handle_training_error(error: Error, logger: PythonObject) raises:
    """Handle training-related errors"""
    if "CUDA out of memory" in str(error):
        logger.error("CUDA out of memory error occurred.")
        logger.error("Consider reducing the batch size or model complexity.")
        raise Error("CUDA out of memory error occurred.")
    else:
        logger.error("A runtime error occurred: " + String(error))
        raise Error("A runtime error occurred: " + String(error))

fn monitor_gpu_memory(args: PythonObject, logger: PythonObject) raises:
    """Monitor GPU memory usage if GPU is enabled"""
    if args.use_gpu:
        var torch = Python.import_module("torch")
        if torch.cuda.is_available():
            var memory_allocated = torch.cuda.memory_allocated()
            var memory_reserved = torch.cuda.memory_reserved()
            logger.info("CUDA memory allocated: " + String(memory_allocated) + " bytes")
            logger.info("CUDA memory reserved: " + String(memory_reserved) + " bytes")


fn init_optuna_config(args: PythonObject, logger: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """Initialize configuration using Optuna for hyperparameter optimization"""
    var optuna = Python.import_module("optuna")
    logger.info("Loading best hyperparameters from Optuna study")
    
    # Get study name
    var study_name = args.optuna_study_name
    if study_name is None:
        study_name = get_default_study_name(args, optuna, logger)
    
    # Load or create study
    try:
        var study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)
    except KeyError:
        var study = optuna.create_study(study_name=study_name, storage=args.optuna_storage)
    
    var best_params = study.best_params
    logger.debug("Loaded best parameters: " + String(best_params))
    
    # Calculate derived parameters
    var n_head = 2 ** best_params["n_head_exp"]
    var n_embd = calculate_embedding_dim(n_head, best_params["n_embd_multiplier"])
    
    # Create configurations
    var ModelConfig = Python.import_module("gpt2_arc.src.config").ModelConfig
    var TrainingConfig = Python.import_module("gpt2_arc.src.config").TrainingConfig
    
    var model_config = ModelConfig(
        n_embd=n_embd,
        n_head=n_head,
        n_layer=best_params["n_layer"],
        dropout=best_params["dropout"],
        num_workers=args.num_workers if args.num_workers is not None else Python.import_module("multiprocessing").cpu_count(),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        pin_memory=not args.no_pin_memory,
    )
    
    var training_config = create_training_config(args, best_params)
    
    return model_config, training_config

fn init_default_config(args: PythonObject, logger: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """Initialize configuration using default or provided parameters"""
    var ModelConfig = Python.import_module("gpt2_arc.src.config").ModelConfig
    var TrainingConfig = Python.import_module("gpt2_arc.src.config").TrainingConfig
    
    var model_config = ModelConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        mamba_ratio=args.mamba_ratio,
        d_state=args.d_state,
        d_conv=args.d_conv,
        mamba_depth=args.mamba_depth,
        mamba_expand=args.mamba_expand,
    )
    
    var training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        use_gpu=args.use_gpu,
        log_level=args.log_level,
        use_synthetic_data=args.use_synthetic_data,
        synthetic_data_path=args.synthetic_data_path,
        use_grokfast=args.use_grokfast,
        grokfast_type=args.grokfast_type,
        grokfast_alpha=args.grokfast_alpha,
        grokfast_lamb=args.grokfast_lamb,
        grokfast_window_size=args.grokfast_window_size,
        include_pad_in_loss=args.include_pad_in_loss,
        include_pad_in_accuracy=args.include_pad_in_accuracy,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        pin_memory=args.pin_memory,
    )
    
    return model_config, training_config

fn calculate_symbol_frequencies(
    args: PythonObject,
    train_data: PythonObject,
    config: PythonObject,
    logger: PythonObject
) raises -> Dict[Int, Float64]:
    """Calculate symbol frequencies for balanced training"""
    if not args.enable_symbol_freq:
        logger.debug("Symbol frequency calculation is disabled. Using empty symbol_freq_dict.")
        return {}
    
    logger.debug("Calculating symbol frequencies as it is enabled.")
    var symbol_freq = train_data.get_symbol_frequencies()
    
    var symbol_freq_dict = {}
    for (i, freq) in symbol_freq.enumerate():
        symbol_freq_dict[i] = Float64(freq)
    
    var pad_symbol_idx = config.training.pad_symbol_idx
    symbol_freq_dict.pop(pad_symbol_idx, None)
    
    logger.debug("Removed pad_symbol_idx (" + String(pad_symbol_idx) + 
                ") from symbol_freq_dict. New length: " + String(len(symbol_freq_dict)))
    
    assert(len(symbol_freq_dict) == config.training.num_classes - 1,
            "Length of symbol_freq_dict (" + String(len(symbol_freq_dict)) + 
            ") does not match num_classes minus padding (" + 
            String(config.training.num_classes - 1) + ").")
    
    return symbol_freq_dict

fn setup_profiler(args: PythonObject) raises -> PythonObject:
    """Setup profiler if enabled"""
    if not args.use_profiler:
        return None
        
    var ProfilerActivity = Python.import_module("torch.profiler").ProfilerActivity
    var profiler = Python.import_module("lightning.pytorch.profilers").PyTorchProfiler(
        dirpath=args.profiler_dirpath,
        filename=args.profiler_filename,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    )
    return profiler

fn determine_accelerator(args: PythonObject) -> Tuple[String, Int, String]:
    """Determine accelerator, devices, and strategy based on args"""
    if args.accelerator == "tpu":
        return "tpu", "xla:1", "tpu_spawn"
    elif args.accelerator == "gpu":
        var torch = Python.import_module("torch")
        if torch.cuda.is_available():
            return "gpu", 1, "auto"
        return "cpu", 1, "auto"
    else:
        return "cpu", 1, "auto"

fn initialize_model(
    config: PythonObject,
    checkpoint_path: String,
    logger: PythonObject
) raises -> PythonObject:
    """Initialize or load model from checkpoint"""
    var torch = Python.import_module("torch")
    var GPT2ARC = Python.import_module("gpt2_arc.src.models.gpt2").GPT2ARC
    
    if checkpoint_path:
        logger.info("Loading model from checkpoint: " + checkpoint_path)
        var checkpoint = torch.load(checkpoint_path)
        validate_checkpoint(checkpoint, logger)
        return load_model_from_checkpoint(checkpoint, config, logger)
    
    logger.info("Initializing new model")
    var model = GPT2ARC(
        config=config,
        num_classes=config.training.num_classes,
        symbol_freq=config.training.symbol_freq,
        pad_symbol_idx=config.training.pad_symbol_idx
    )
    
    # Calculate layer distribution
    var mamba_layers = Int(config.model.n_layer * config.model.mamba_ratio)
    var transformer_layers = config.model.n_layer - mamba_layers
    validate_layer_distribution(mamba_layers, transformer_layers, config, logger)
    
    return model

fn process_test_results(
    test_results: List[Dict[String, Float64]], 
    logger: PythonObject
) -> Dict[String, Float64]:
    """Process and aggregate test results"""
    var sum_test_loss = 0.0
    var sum_test_accuracy = 0.0
    var sum_test_diff_accuracy = 0.0
    
    for result in test_results:
        sum_test_loss += result["avg_test_loss"]
        sum_test_accuracy += result["avg_test_accuracy"]
        sum_test_diff_accuracy += result["avg_test_diff_accuracy"]
    
    var avg_test_loss = sum_test_loss / Float64(len(test_results))
    var avg_test_accuracy = sum_test_accuracy / Float64(len(test_results))
    var avg_test_diff_accuracy = sum_test_diff_accuracy / Float64(len(test_results))
    
    logger.info("Test results - Loss: " + String(avg_test_loss) + 
                ", Accuracy: " + String(avg_test_accuracy) + 
                ", Diff Accuracy: " + String(avg_test_diff_accuracy))
    
    # Compile results dictionary
    var results = {
        "avg_test_loss": avg_test_loss,
        "avg_test_accuracy": avg_test_accuracy,
        "avg_test_diff_accuracy": avg_test_diff_accuracy
    }
    
    # Add task-specific results
    for result in test_results:
        for key, value in result.items():
            if key.endswith("_test_accuracy") or key.endswith("_test_diff_accuracy"):
                results[key] = value
    
    return results

fn handle_training_error(error: Error, logger: PythonObject) raises:
    """Handle training-related errors"""
    if "CUDA out of memory" in str(error):
        logger.error("CUDA out of memory error occurred.")
        logger.error("Consider reducing the batch size or model complexity.")
        raise Error("CUDA out of memory error occurred.")
    else:
        logger.error("A runtime error occurred: " + String(error))
        raise Error("A runtime error occurred: " + String(error))

fn monitor_gpu_memory(args: PythonObject, logger: PythonObject) raises:
    """Monitor GPU memory usage if GPU is enabled"""
    if args.use_gpu:
        var torch = Python.import_module("torch")
        if torch.cuda.is_available():
            var memory_allocated = torch.cuda.memory_allocated()
            var memory_reserved = torch.cuda.memory_reserved()
            logger.info("CUDA memory allocated: " + String(memory_allocated) + " bytes")
            logger.info("CUDA memory reserved: " + String(memory_reserved) + " bytes")

fn train_model(args: PythonObject) raises -> None:
    """Main training orchestration function"""
    var logging = Python.import_module("logging")
    var logger = logging.getLogger("train")
    logger.setLevel(Python.evaluate("logging.DEBUG"))
    
    try:
        # Initialize configuration
        var model_config, training_config = init_config(args, logger)
        
        # Setup trainer with all components
        var trainer = setup_trainer(args, logger)
        
        # Load datasets
        var train_data, val_data, test_data = load_datasets(args, config, logger)
        
        # Calculate symbol frequencies if enabled
        var symbol_freq_dict = calculate_symbol_frequencies(args, train_data, config, logger)
        
        # Update config with symbol frequencies
        config = update_config_with_frequencies(config, symbol_freq_dict, args)
        
        # Initialize model
        var model = initialize_model(config, args.model_checkpoint, logger)
        
        # Monitor initial GPU memory
        monitor_gpu_memory(args, logger)
        
        # Train and evaluate
        var results = train_and_evaluate(trainer, model, train_data, val_data, test_data, args, logger)
        
        # Monitor final GPU memory
        monitor_gpu_memory(args, logger)
        
        # Save model and results
        save_model_and_results(model, results, model.results_collector.experiment_id, logger)
        
        logger.info("Training completed successfully")
        
    except Error as e:
        handle_training_error(e, logger)
        raise
    finally:
        if "tracker" in locals():
            tracker.finish()