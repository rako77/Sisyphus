using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection.PortableExecutable;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;

namespace DupExtractor;

/// <summary>
/// Walks a C# source tree, builds one Roslyn compilation per project (so locals, parameters
/// and fields resolve through the semantic model), and writes one JSON line per method-like
/// chunk: code, normalized token stream, hashes, size metrics and noise flags.
/// </summary>
public static class Program
{
    private const string ImplicitUsingsPath = "__dupextractor_implicit_usings.cs";

    // Mirrors the SDK's <ImplicitUsings> so BCL types resolve in projects that rely on them.
    private const string ImplicitUsings = """
        global using System;
        global using System.Collections.Generic;
        global using System.IO;
        global using System.Linq;
        global using System.Net.Http;
        global using System.Threading;
        global using System.Threading.Tasks;
        """;

    public static int Main(string[] args)
    {
        Options? opts;
        try { opts = Options.Parse(args); }
        catch (Exception e) { Console.Error.WriteLine(e.Message); Options.PrintUsage(); return 2; }
        if (opts is null) { Options.PrintUsage(); return 2; }

        var root = Path.GetFullPath(opts.Root);
        if (!Directory.Exists(root)) { Console.Error.WriteLine($"Not a directory: {root}"); return 2; }

        var sw = Stopwatch.StartNew();
        var files = FileDiscovery.Find(root, opts.ExcludeDirs);
        Console.Error.WriteLine($"Found {files.Count} .cs files under {root}");

        var references = FrameworkReferences.Load();
        var parseOptions = new CSharpParseOptions(LanguageVersion.Preview, preprocessorSymbols: opts.Defines);
        var compilationOptions = new CSharpCompilationOptions(
            OutputKind.DynamicallyLinkedLibrary,
            allowUnsafe: true,
            nullableContextOptions: NullableContextOptions.Enable,
            concurrentBuild: true);

        var chunks = new ConcurrentBag<Chunk>();
        int skippedGenerated = 0;

        foreach (var group in files.GroupBy(f => FileDiscovery.ProjectFor(f, root)).OrderBy(g => g.Key.Name, StringComparer.Ordinal))
        {
            var projectName = group.Key.Name;
            var trees = group.AsParallel().Select(path =>
            {
                using var stream = File.OpenRead(path);
                var text = SourceText.From(stream);
                return CSharpSyntaxTree.ParseText(text, parseOptions, path);
            }).ToList();
            trees.Add(CSharpSyntaxTree.ParseText(ImplicitUsings, parseOptions, ImplicitUsingsPath));

            // Generated files stay in the compilation (other code references their types);
            // they are only excluded from extraction.
            var compilation = CSharpCompilation.Create(projectName, trees, references, compilationOptions);

            int before = chunks.Count;
            Parallel.ForEach(trees, tree =>
            {
                if (tree.FilePath == ImplicitUsingsPath) return;
                bool generatedFile = GeneratedDetector.IsGeneratedFile(tree.FilePath, tree.GetText());
                if (generatedFile && !opts.IncludeGenerated)
                {
                    Interlocked.Increment(ref skippedGenerated);
                    return;
                }
                var model = compilation.GetSemanticModel(tree);
                foreach (var chunk in ChunkExtractor.Extract(tree, model, root, projectName, generatedFile, opts))
                    chunks.Add(chunk);
            });
            Console.Error.WriteLine($"  {projectName}: {trees.Count - 1} files, {chunks.Count - before} chunks");
        }

        var ordered = chunks
            .OrderBy(c => c.File, StringComparer.Ordinal)
            .ThenBy(c => c.StartLine)
            .ThenBy(c => c.Id, StringComparer.Ordinal)
            .ToList();

        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };
        var outDir = Path.GetDirectoryName(Path.GetFullPath(opts.Output));
        if (!string.IsNullOrEmpty(outDir)) Directory.CreateDirectory(outDir);
        using (var writer = new StreamWriter(opts.Output, append: false, new UTF8Encoding(false)))
        {
            foreach (var chunk in ordered)
                writer.WriteLine(JsonSerializer.Serialize(chunk, jsonOptions));
        }

        var byKind = string.Join(", ", ordered.GroupBy(c => c.Kind).OrderBy(g => g.Key).Select(g => $"{g.Key}={g.Count()}"));
        Console.Error.WriteLine($"Wrote {ordered.Count} chunks ({byKind}) to {opts.Output}");
        Console.Error.WriteLine($"Skipped {skippedGenerated} generated files. Took {sw.Elapsed.TotalSeconds:F1}s.");
        return 0;
    }
}

internal sealed class Options
{
    public string Root = "";
    public string Output = "chunks.jsonl";
    public bool IncludeGenerated;
    public int MinTokens = 12;
    public List<string> ExcludeDirs = new();
    public string[] Defines = { "DEBUG", "TRACE" };

    public static Options? Parse(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"Missing value for {a}");
            switch (a)
            {
                case "-o":
                case "--out": o.Output = Next(); break;
                case "--include-generated": o.IncludeGenerated = true; break;
                case "--min-tokens": o.MinTokens = int.Parse(Next()); break;
                case "--exclude-dir": o.ExcludeDirs.Add(Next()); break;
                case "--define": o.Defines = Next().Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries); break;
                case "-h":
                case "--help": return null;
                default:
                    if (a.StartsWith('-')) throw new ArgumentException($"Unknown option {a}");
                    if (o.Root.Length > 0) throw new ArgumentException("Only one root directory may be given");
                    o.Root = a;
                    break;
            }
        }
        return o.Root.Length == 0 ? null : o;
    }

    public static void PrintUsage()
    {
        Console.Error.WriteLine("""
            Usage: DupExtractor <source-root> [options]

              -o, --out <file>        Output JSONL path (default: chunks.jsonl)
              --min-tokens <n>        Skip chunks with fewer normalized tokens (default: 12)
              --exclude-dir <name>    Skip directories with this name (repeatable).
                                      bin, obj, .git, .vs, .idea, node_modules, TestResults are always skipped.
              --include-generated     Also extract generated code (flagged 'generated')
              --define <A,B>          Preprocessor symbols (default: DEBUG,TRACE)
            """);
    }
}

internal static class FileDiscovery
{
    private static readonly string[] AlwaysSkip = { "bin", "obj", ".git", ".vs", ".idea", "node_modules", "TestResults" };
    private static readonly ConcurrentDictionary<string, (string Name, string Dir)> Cache = new(StringComparer.Ordinal);

    public static List<string> Find(string root, IEnumerable<string> extraExcludes)
    {
        var skip = new HashSet<string>(AlwaysSkip, StringComparer.OrdinalIgnoreCase);
        foreach (var e in extraExcludes) skip.Add(e);

        var result = new List<string>();
        var stack = new Stack<string>();
        stack.Push(root);
        while (stack.Count > 0)
        {
            var dir = stack.Pop();
            try
            {
                result.AddRange(Directory.EnumerateFiles(dir, "*.cs"));
                foreach (var sub in Directory.EnumerateDirectories(dir))
                    if (!skip.Contains(Path.GetFileName(sub))) stack.Push(sub);
            }
            catch (UnauthorizedAccessException) { }
            catch (IOException) { }
        }
        result.Sort(StringComparer.Ordinal);
        return result;
    }

    /// <summary>Nearest ancestor directory (within root) containing a .csproj.</summary>
    public static (string Name, string Dir) ProjectFor(string file, string root) =>
        Lookup(Path.GetDirectoryName(file)!, root);

    private static (string Name, string Dir) Lookup(string dir, string root) => Cache.GetOrAdd(dir, d =>
    {
        var proj = Directory.EnumerateFiles(d, "*.csproj").OrderBy(x => x, StringComparer.Ordinal).FirstOrDefault();
        if (proj != null) return (Path.GetFileNameWithoutExtension(proj), d);
        var parent = Path.GetDirectoryName(d);
        if (parent == null || string.Equals(Path.TrimEndingDirectorySeparator(d), Path.TrimEndingDirectorySeparator(root), StringComparison.Ordinal))
            return ("(no project)", root);
        return Lookup(parent, root);
    });
}

internal static class FrameworkReferences
{
    /// <summary>The running runtime's framework assemblies (System.*, etc.).</summary>
    public static List<MetadataReference> Load()
    {
        var tpa = AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string ?? "";
        var runtimeDir = Path.GetDirectoryName(typeof(object).Assembly.Location) ?? "";
        return tpa.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Where(p => p.StartsWith(runtimeDir, StringComparison.OrdinalIgnoreCase)
                        && p.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)
                        && IsManaged(p))
            .Select(p => (MetadataReference)MetadataReference.CreateFromFile(p))
            .ToList();
    }

    private static bool IsManaged(string path)
    {
        try
        {
            using var fs = File.OpenRead(path);
            using var pe = new PEReader(fs);
            return pe.HasMetadata;
        }
        catch { return false; }
    }
}

internal static class GeneratedDetector
{
    private static readonly string[] Suffixes = { ".g.cs", ".g.i.cs", ".designer.cs", ".generated.cs", ".AssemblyInfo.cs", ".AssemblyAttributes.cs" };

    public static bool IsGeneratedFile(string path, SourceText text)
    {
        var name = Path.GetFileName(path);
        if (Suffixes.Any(s => name.EndsWith(s, StringComparison.OrdinalIgnoreCase))) return true;
        if (name.Equals("AssemblyInfo.cs", StringComparison.OrdinalIgnoreCase)) return true;

        var normalized = path.Replace('\\', '/');
        if (normalized.Contains("/Migrations/", StringComparison.OrdinalIgnoreCase)) return true;

        var head = text.ToString(new TextSpan(0, Math.Min(2000, text.Length)));
        return head.Contains("<auto-generated", StringComparison.OrdinalIgnoreCase)
            || head.Contains("<autogenerated", StringComparison.OrdinalIgnoreCase);
    }
}
