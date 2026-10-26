using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace DupExtractor;

/// <summary>One line of the output JSONL. Field names are camelCased on output.</summary>
public sealed record Chunk(
    string Id,
    string File,
    string Project,
    string Namespace,
    string Type,
    string Member,
    string Kind,
    string Signature,
    int StartLine,
    int EndLine,
    int CodeLines,
    int Tokens,
    int Statements,
    string? ParentId,
    string[] Flags,
    string[] Hints,
    string ExactHash,
    string NormalizedHash,
    string Code,
    string Normalized);

internal static class ChunkExtractor
{
    private static readonly HashSet<string> TestAttributes = new(StringComparer.Ordinal)
    {
        "Fact", "Theory", "Test", "TestMethod", "DataTestMethod", "TestCase", "TestCaseSource",
        "SetUp", "TearDown", "OneTimeSetUp", "OneTimeTearDown", "TestInitialize", "TestCleanup",
        "ClassInitialize", "ClassCleanup", "TestFixture", "TestClass",
    };

    private static readonly Regex TestProject = new(@"(?i)(tests?|specs?|\.testing)$", RegexOptions.Compiled);
    private static readonly Regex MappingName = new(@"^(Map|To[A-Z]|From[A-Z]|Convert|Project[A-Z])", RegexOptions.Compiled);
    private static readonly Regex RepositoryType = new(@"(Repository|Repo|Store|Dao|DataAccess)(<.*>)?$", RegexOptions.Compiled);

    private static readonly HashSet<string> PropertyChangedCalls = new(StringComparer.Ordinal)
    {
        "OnPropertyChanged", "RaisePropertyChanged", "NotifyPropertyChanged", "OnPropertyChanging",
        "SetProperty", "SetField", "Set", "RaiseAndSetIfChanged", "NotifyOfPropertyChange",
    };

    private static readonly HashSet<string> DiReceiverTypes = new(StringComparer.Ordinal)
    {
        "IServiceCollection", "IApplicationBuilder", "WebApplicationBuilder", "WebApplication", "IHostBuilder",
        "IHostApplicationBuilder", "IEndpointRouteBuilder", "ContainerBuilder", "IMvcBuilder",
    };

    public static IEnumerable<Chunk> Extract(SyntaxTree tree, SemanticModel model, string root, string project, bool generatedFile, Options opts)
    {
        var text = tree.GetText();
        var rel = Path.GetRelativePath(root, tree.FilePath).Replace('\\', '/');
        var fileName = Path.GetFileName(tree.FilePath);
        bool startupFile = fileName is "Program.cs" or "Startup.cs";
        bool testPath = TestProject.IsMatch(project)
            || rel.Split('/').SkipLast(1).Any(seg => TestProject.IsMatch(seg) && seg.Length <= 30);

        foreach (var node in tree.GetRoot().DescendantNodes())
        {
            var site = SiteFactory.Describe(node);
            if (site is null) continue;

            var attrs = AttributeNames(node);
            bool generated = generatedFile || attrs.Contains("GeneratedCode") || attrs.Contains("CompilerGenerated");
            if (generated && !opts.IncludeGenerated) continue;

            var normalizer = new Normalizer(model, FieldNames(node), TypeParameterNames(node));
            var tokens = normalizer.Run(site);
            if (tokens.Count < opts.MinTokens) continue;

            var lineSpan = node.GetLocation().GetLineSpan();
            int startLine = lineSpan.StartLinePosition.Line + 1;
            int endLine = lineSpan.EndLinePosition.Line + 1;

            var parentSite = node.Ancestors().Select(SiteFactory.Describe).FirstOrDefault(s => s != null);
            string member = site.Kind switch
            {
                "lambda" => $"{parentSite?.Name ?? "?"}.<lambda@L{startLine}>",
                "localFunction" => $"{parentSite?.Name ?? "?"}.{site.Name}",
                _ => site.Name,
            };

            var code = ExtractCode(text, node);
            var core = CoreStatements(site, normalizer.GuardCount);
            var (flags, hints) = Classify(site, core, attrs, node, testPath, startupFile, generated);
            var normalized = string.Join(' ', tokens);

            yield return new Chunk(
                Id: IdFor(rel, node, site.Kind),
                File: rel,
                Project: project,
                Namespace: NamespaceOf(node),
                Type: TypeChain(node),
                Member: member,
                Kind: site.Kind,
                Signature: site.Signature,
                StartLine: startLine,
                EndLine: endLine,
                CodeLines: CountCodeLines(code),
                Tokens: tokens.Count,
                Statements: normalizer.CountStatements(site),
                ParentId: parentSite is null ? null : IdFor(rel, parentSite.Node, parentSite.Kind),
                Flags: flags,
                Hints: hints,
                ExactHash: Hash(string.Join(' ', Normalizer.RawTokens(site))),
                NormalizedHash: Hash(normalized),
                Code: code,
                Normalized: normalized);
        }
    }

    // ---- classification ------------------------------------------------------------------

    private static (string[] Flags, string[] Hints) Classify(
        Site site, IReadOnlyList<StatementSyntax> core, HashSet<string> attrs, SyntaxNode node,
        bool testPath, bool startupFile, bool generated)
    {
        var flags = new List<string>();
        var hints = new List<string>();
        string baseName = site.Name.Split('<')[0];

        if (generated) flags.Add("generated");
        if (testPath || attrs.Overlaps(TestAttributes) || TypeChain(node).Split('.').Any(t => t.EndsWith("Tests", StringComparison.Ordinal)))
            flags.Add("test");

        if (site.Block != null && core.Count == 0) flags.Add("empty");
        if (IsForwarder(site, core)) flags.Add("forwarder");
        if (IsTrivialConstructor(site, core)) flags.Add("trivialCtor");

        if (baseName is "Equals" or "GetHashCode" or "CompareTo"
            || (site.Kind == "operator" && site.Name is "operator ==" or "operator !=" or "operator <" or "operator >" or "operator <=" or "operator >="))
            flags.Add("equality");
        if (baseName == "ToString" && site.Kind == "method") flags.Add("toString");
        if (baseName is "Dispose" or "DisposeAsync" or "DisposeCore" or "DisposeAsyncCore" or "Finalize") flags.Add("dispose");

        if (IsPropertyChangedPlumbing(site, baseName)) flags.Add("propertyChanged");
        if (IsDiWiring(site, baseName)) flags.Add("diWiring");
        if (startupFile) flags.Add("startup");

        if (MappingName.IsMatch(baseName) && HasLargeInitializer(node)) hints.Add("mapping");
        if (node.Ancestors().OfType<TypeDeclarationSyntax>().FirstOrDefault() is { } type && RepositoryType.IsMatch(type.Identifier.ValueText))
            hints.Add("repository");
        if (node.DescendantNodes().OfType<QueryExpressionSyntax>().Any()) hints.Add("linqQuery");

        return (flags.ToArray(), hints.ToArray());
    }

    private static IReadOnlyList<StatementSyntax> CoreStatements(Site site, int guardCount) =>
        site.Block?.Statements.Skip(guardCount).ToList() ?? (IReadOnlyList<StatementSyntax>)Array.Empty<StatementSyntax>();

    /// <summary>A body that is one call whose arguments are just passed through.</summary>
    private static bool IsForwarder(Site site, IReadOnlyList<StatementSyntax> core)
    {
        if (site.Node is ConstructorDeclarationSyntax ctor)
            return ctor.Initializer != null && site.Block != null && core.Count == 0;

        ExpressionSyntax? e = site.Arrow?.Expression;
        if (e == null && core.Count == 1)
        {
            e = core[0] switch
            {
                ReturnStatementSyntax r => r.Expression,
                ExpressionStatementSyntax es => es.Expression,
                _ => null,
            };
        }
        if (e is AwaitExpressionSyntax aw) e = aw.Expression;
        if (e is InvocationExpressionSyntax { Expression: MemberAccessExpressionSyntax ma } && ma.Name.Identifier.ValueText == "ConfigureAwait")
            e = ma.Expression;
        return e is InvocationExpressionSyntax inv && inv.ArgumentList.Arguments.All(a => IsTrivialArgument(a.Expression));
    }

    private static bool IsTrivialArgument(ExpressionSyntax e) => e switch
    {
        IdentifierNameSyntax or LiteralExpressionSyntax or DefaultExpressionSyntax or ThisExpressionSyntax => true,
        MemberAccessExpressionSyntax ma => ma.Expression is IdentifierNameSyntax or ThisExpressionSyntax,
        PostfixUnaryExpressionSyntax p when p.IsKind(SyntaxKind.SuppressNullableWarningExpression) => IsTrivialArgument(p.Operand),
        CastExpressionSyntax c => IsTrivialArgument(c.Expression),
        _ => false,
    };

    /// <summary>A constructor that only copies arguments into fields/properties.</summary>
    private static bool IsTrivialConstructor(Site site, IReadOnlyList<StatementSyntax> core)
    {
        if (site.Node is not ConstructorDeclarationSyntax) return false;
        if (site.Arrow != null) return site.Arrow.Expression is AssignmentExpressionSyntax a && IsSimpleAssignment(a);
        return core.Count > 0 && core.All(st => st is ExpressionStatementSyntax { Expression: AssignmentExpressionSyntax a } && IsSimpleAssignment(a));
    }

    private static bool IsSimpleAssignment(AssignmentExpressionSyntax a)
    {
        if (!a.IsKind(SyntaxKind.SimpleAssignmentExpression)) return false;
        var right = a.Right;
        if (right is BinaryExpressionSyntax b && b.IsKind(SyntaxKind.CoalesceExpression) && b.Right is ThrowExpressionSyntax)
            right = b.Left;
        return right switch
        {
            IdentifierNameSyntax or LiteralExpressionSyntax => true,
            TupleExpressionSyntax t => t.Arguments.All(x => x.Expression is IdentifierNameSyntax),
            ObjectCreationExpressionSyntax oc => (oc.ArgumentList == null || oc.ArgumentList.Arguments.Count == 0) && oc.Initializer == null,
            ImplicitObjectCreationExpressionSyntax io => io.ArgumentList.Arguments.Count == 0 && io.Initializer == null,
            CollectionExpressionSyntax ce => ce.Elements.Count == 0,
            _ => false,
        };
    }

    private static bool IsPropertyChangedPlumbing(Site site, string baseName)
    {
        if (site.Kind == "method")
            return PropertyChangedCalls.Contains(baseName) && site.Node.ToString().Contains("PropertyChang", StringComparison.Ordinal);
        if (site.Kind != "accessor" || !(site.Name.EndsWith(".set") || site.Name.EndsWith(".init"))) return false;
        return site.Node.DescendantNodes().OfType<InvocationExpressionSyntax>().Any(inv =>
        {
            var name = inv.Expression switch
            {
                IdentifierNameSyntax id => id.Identifier.ValueText,
                GenericNameSyntax g => g.Identifier.ValueText,
                MemberAccessExpressionSyntax ma => ma.Name.Identifier.ValueText,
                MemberBindingExpressionSyntax mb => mb.Name.Identifier.ValueText,
                _ => "",
            };
            return PropertyChangedCalls.Contains(name) || name == "Invoke" && inv.ToString().Contains("PropertyChanged", StringComparison.Ordinal);
        });
    }

    private static bool IsDiWiring(Site site, string baseName)
    {
        if (site.Kind != "method") return false;
        if (baseName is "ConfigureServices" or "ConfigureContainer" or "RegisterServices" or "AddServices") return true;
        if (site.Node is MethodDeclarationSyntax md && md.ParameterList.Parameters.FirstOrDefault()?.Type is { } firstType)
        {
            var typeName = firstType switch
            {
                QualifiedNameSyntax q => q.Right.Identifier.ValueText,
                SimpleNameSyntax s => s.Identifier.ValueText,
                _ => firstType.ToString(),
            };
            return DiReceiverTypes.Contains(typeName);
        }
        return false;
    }

    private static bool HasLargeInitializer(SyntaxNode node) =>
        node.DescendantNodes().OfType<InitializerExpressionSyntax>()
            .Any(i => i.IsKind(SyntaxKind.ObjectInitializerExpression) && i.Expressions.Count >= 3)
        || node.DescendantNodes().OfType<AnonymousObjectCreationExpressionSyntax>().Any(a => a.Initializers.Count >= 3);

    // ---- naming / location ---------------------------------------------------------------

    private static HashSet<string> AttributeNames(SyntaxNode node)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        foreach (var n in node.AncestorsAndSelf())
        {
            SyntaxList<AttributeListSyntax> lists = n switch
            {
                MemberDeclarationSyntax m => m.AttributeLists,
                LocalFunctionStatementSyntax lf => lf.AttributeLists,
                AccessorDeclarationSyntax a => a.AttributeLists,
                _ => default,
            };
            foreach (var list in lists)
            foreach (var attr in list.Attributes)
            {
                var name = attr.Name switch
                {
                    QualifiedNameSyntax q => q.Right.Identifier.ValueText,
                    AliasQualifiedNameSyntax al => al.Name.Identifier.ValueText,
                    SimpleNameSyntax s => s.Identifier.ValueText,
                    _ => attr.Name.ToString(),
                };
                if (name.EndsWith("Attribute", StringComparison.Ordinal) && name.Length > "Attribute".Length)
                    name = name[..^"Attribute".Length];
                set.Add(name);
            }
        }
        return set;
    }

    private static HashSet<string> FieldNames(SyntaxNode node)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        var type = node.Ancestors().OfType<TypeDeclarationSyntax>().FirstOrDefault();
        if (type == null) return set;
        foreach (var field in type.Members.OfType<FieldDeclarationSyntax>())
        {
            if (field.Modifiers.Any(SyntaxKind.ConstKeyword)) continue;
            foreach (var v in field.Declaration.Variables) set.Add(v.Identifier.ValueText);
        }
        return set;
    }

    private static HashSet<string> TypeParameterNames(SyntaxNode node)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        foreach (var n in node.AncestorsAndSelf())
        {
            var list = n switch
            {
                MethodDeclarationSyntax m => m.TypeParameterList,
                LocalFunctionStatementSyntax lf => lf.TypeParameterList,
                TypeDeclarationSyntax t => t.TypeParameterList,
                _ => null,
            };
            if (list != null)
                foreach (var tp in list.Parameters) set.Add(tp.Identifier.ValueText);
        }
        return set;
    }

    private static string TypeChain(SyntaxNode node)
    {
        var parts = node.Ancestors().OfType<BaseTypeDeclarationSyntax>().Reverse()
            .Select(t => t.Identifier.ValueText + (t is TypeDeclarationSyntax { TypeParameterList: { } tp } ? tp.ToString() : ""));
        var chain = string.Join('.', parts);
        return chain.Length == 0 ? "(global)" : chain;
    }

    private static string NamespaceOf(SyntaxNode node)
    {
        var parts = node.Ancestors().OfType<BaseNamespaceDeclarationSyntax>().Reverse().Select(n => n.Name.ToString());
        return string.Join('.', parts);
    }

    private static string IdFor(string rel, SyntaxNode node, string kind)
    {
        var pos = node.GetLocation().GetLineSpan().StartLinePosition;
        return $"{rel}:{pos.Line + 1}:{pos.Character + 1}:{kind}";
    }

    /// <summary>The node's source, starting at the beginning of its first line, dedented.</summary>
    private static string ExtractCode(SourceText text, SyntaxNode node)
    {
        var firstLine = text.Lines.GetLineFromPosition(node.SpanStart);
        // Keep the real indentation of the first line but blank out any code that precedes the node on it.
        var lead = text.ToString(TextSpan.FromBounds(firstLine.Start, node.SpanStart));
        var leadWs = new string(lead.Select(ch => ch == '\t' ? '\t' : ' ').ToArray());
        var raw = leadWs + text.ToString(node.Span);
        var lines = raw.Replace("\r\n", "\n").Split('\n');
        int indent = lines.Where(l => l.Trim().Length > 0)
            .Select(l => l.Length - l.TrimStart().Length)
            .DefaultIfEmpty(0)
            .Min();
        return string.Join('\n', lines.Select(l => l.Length >= indent && l[..indent].Trim().Length == 0 ? l[indent..] : l.TrimStart()));
    }

    private static int CountCodeLines(string code) =>
        code.Split('\n').Count(l =>
        {
            var t = l.Trim();
            return t.Length > 0 && !t.StartsWith("//", StringComparison.Ordinal) && !t.StartsWith("/*", StringComparison.Ordinal)
                && !t.StartsWith("*", StringComparison.Ordinal) && t is not "{" and not "}";
        });

    private static string Hash(string s) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(s)))[..32].ToLowerInvariant();
}
