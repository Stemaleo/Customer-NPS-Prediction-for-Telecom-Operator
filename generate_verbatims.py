import pandas as pd
import numpy as np

def generate_verbatims(df_path, output_path, seed=42):
    np.random.seed(seed)
    df = pd.read_csv(df_path)
    
    verbatims = []
    
    # Templates for Detractors (Satisfaction 1, 2, or 3-churned)
    detractor_templates = [
        "Customer called to complain about constant {service} drops. They are extremely frustrated with the lack of reliable connection and threatened to cancel.",
        "Agent notes: customer is angry about the sudden price hike. They mentioned that competitors offer fiber for cheaper and they want a contract review.",
        "Frustrated chat transcript: 'My {service} has been down twice this week. I work from home and this is unacceptable. I want a refund.'",
        "App review: 'Worst service ever. The billing is inconsistent and customer support keeps transferring me without solving my issue.'",
        "Customer is complaining about poor internet speed. They said they pay for fiber but get speeds slower than DSL. Very unhappy.",
        "Billing dispute: Customer claims they were charged extra for services they didn't authorize. They are demanding immediate correction.",
        "Tech support note: Customer had 3 technician visits this month for {service} issues and it is still not fixed. Customer is highly dissatisfied."
    ]
    
    # Templates for Passives (Satisfaction 4, or 3-stayed)
    passive_templates = [
        "Customer called for routine billing query. They seemed satisfied with the explanation and had no other issues.",
        "Chat transcript: customer asked about updating their payment method. Service is working fine, no major complaints.",
        "Agent notes: customer asked if there are any loyalty discounts available. Overall neutral feedback about the {service}.",
        "Customer is satisfied with the {service} speed but feels the monthly cost is slightly high compared to new customer promotions.",
        "Support note: assisted customer with resetting their password. Service is running normally.",
        "Customer mentioned that the connection is generally stable, but they occasionally experience slow loading during peak hours.",
        "Chat transcript: customer inquired about upgrading their TV package. Neutral tone throughout."
    ]
    
    # Templates for Promoters (Satisfaction 5)
    promoter_templates = [
        "Customer called to express how happy they are with the new fiber installation. Mentioned the speed is fantastic and the tech was very professional.",
        "Agent notes: Customer is highly satisfied. They love the reliable {service} and said they always recommend us to their friends.",
        "Chat transcript: 'Just wanted to say thank you to the support team for resolving my query so fast. Great customer service!'",
        "App review: 'Been a customer for years and the connection has never let me down. Highly recommend!'",
        "Customer is extremely happy with the tech support experience. They said the agent went above and beyond to help them.",
        "Support note: Customer is satisfied. They praised the paperless billing and automatic payment setup for being seamless.",
        "Customer called to add another line and commented on how reliable our network has been throughout their tenure."
    ]
    
    # Noise templates (for counter-intuitive cases)
    neutral_noise = "Customer called to check if paperless billing is active. No comments on service quality."
    happy_noise = "Customer is very happy with the customer representative who assisted them today."
    angry_noise = "Customer expressed irritation about a long wait time to reach an agent."

    for idx, row in df.iterrows():
        sat = row['Satisfaction Score']
        churn = row.get('Churn Label', 'No')
        
        # Determine service proxy for templates
        internet_type = row.get('Internet Type', 'Internet')
        if pd.isna(internet_type) or internet_type == 'None':
            service = "phone connection"
        else:
            service = f"{internet_type} internet"
            
        # Determine category base
        if sat <= 2 or (sat == 3 and churn == 'Yes'):
            category = 'detractor'
        elif sat == 5:
            category = 'promoter'
        else:
            category = 'passive'
            
        # Add realistic noise (10% chance of mismatched verbatim tone)
        rand_val = np.random.rand()
        if rand_val < 0.10:  # 10% noise
            if category == 'detractor':
                # Detractor with a neutral/happy note (e.g. they complain but are polite, or it's a simple request)
                text = np.random.choice([neutral_noise, "Customer checked their billing balance. Seemed satisfied with the response."])
            elif category == 'promoter':
                # Promoter with a minor complaint (e.g. unhappy with wait time but likes service)
                text = np.random.choice([angry_noise, "Customer had a minor issue with the app login but loves the service overall."])
            else:
                # Passive with a strong emotion
                text = np.random.choice(detractor_templates + promoter_templates)
        else:
            # Match category
            if category == 'detractor':
                text = np.random.choice(detractor_templates)
            elif category == 'promoter':
                text = np.random.choice(promoter_templates)
            else:
                text = np.random.choice(passive_templates)
                
        # Format service name if present in template
        text = text.format(service=service)
        verbatims.append({
            'Customer ID': row['Customer ID'],
            'Customer Verbatim': text
        })
        
    out_df = pd.DataFrame(verbatims)
    out_df.to_csv(output_path, index=False)
    print(f"Generated {len(out_df)} customer verbatims and saved to {output_path}")

if __name__ == '__main__':
    generate_verbatims('telco_11_1_3.csv', 'customer_verbatims.csv')
